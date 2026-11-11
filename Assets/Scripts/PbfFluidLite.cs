using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// PBF-lite fluid layered on top of Physics Core 2D, Burst + Jobs.
///
/// WHO DOES WHAT
///   Physics Core 2D (engine): integration, gravity (body.gravityScale), damping, mass, shape contacts,
///                             walls / other bodies / Finger, sleeping, sub-stepping.
///   This script:              the one thing the engine has no concept of: the inter-particle density
///                             constraint (pressure) and viscosity.
///
/// PER FIXED STEP (runs right before the engine step)
///   main thread : read positions + velocities in two batched engine calls            -> NativeArrays
///   Burst jobs  : predict -> static-grid bucket -> [density/lambda -> correction -> re-bucket] x iterations
///                 -> constraint velocity -> viscosity + force
///   main thread : hand the forces back as ONE batched force (F = m * dv / dt)
///   The engine integrates them together with gravity and contacts. We never write position or velocity.
///
/// REQUIRES   Burst package (com.unity.burst; it brings Mathematics).
/// ASSUMES    The default world simulates in FixedUpdate (the default): one engine step per FixedUpdate,
///            user forces cleared after each step.
/// SCRIPTING SYMBOL  PBF_PER_BODY_READ  switches the batched engine reads to per-body reads (slower, but needs
///            only APIs that exist in every Unity.U2D.Physics version).
/// </summary>
[DefaultExecutionOrder(10)]
public class PbfFluidLite : MonoBehaviour
{
    [System.Serializable]
    public class FluidType
    {
        public string name = "Water";
        [Tooltip("Absolute rest density. Only used when Auto Calibrate is OFF.")]
        public float restDensity = 20f;
        [Tooltip("Rest density relative to the calibrated lattice (Auto Calibrate ON). Below 1 = looser fluid.")]
        [Range(0.5f, 1.5f)] public float restDensityScale = 1f;
        [Range(0f, 1f)] public float stiffness = 0.8f;
        [Range(0f, 1f)] public float viscosity = 0.2f;
        [Tooltip("Multiplied into the body's own gravityScale. The ENGINE applies gravity.")]
        public float gravityMultiplier = 1f;
    }

    // ------------------------------------------------------------------ inspector
    [Header("Smoothing radius (kernel size)")]
    [Tooltip("ON: radius = Reference Spacing x ratio, so changing the spawner's spacing re-scales the whole solver. " +
             "OFF: use Smoothing Radius below.")]
    public bool smoothingFromSpacing = true;
    [Tooltip("About 2 = ~8 neighbours per particle. Lower = cheaper but lumpier.")]
    public float smoothingToSpacing = 2f;
    public float smoothingRadius = 0.8f;
    public float EffectiveSmoothing =>
        Mathf.Max(0.05f, smoothingFromSpacing ? referenceSpacing * smoothingToSpacing : smoothingRadius);

    [Header("Grid (STATIC. Allocated once; cell size = smoothing radius)")]
    [Tooltip("World position of the grid's bottom-left corner.")]
    public Vector2 gridOrigin = new Vector2(-10f, -10f);
    [Min(1)] public int gridColumns = 32;
    [Min(1)] public int gridRows = 32;
    [Tooltip("Particles outside the grid are clamped into the border cells: they keep working, but get slower if many pile up there.")]
    public bool showGridGizmo = true;
    public bool drawGridWhenUnselected = true;

    [Header("Solver")]
    [Range(1, 10)] public int iterations = 3;
    public bool enablePBF = true;
    [Tooltip("Untick to suspend the fluid pass. The engine keeps simulating the bodies as plain circles.")]
    public bool runInFixedUpdate = true;
    [Tooltip("Cap on how fast the fluid pass may ACCELERATE a particle (m/s).")]
    public float maxSpeed = 20f;
    [Tooltip("Cap on the positional correction per iteration, as a fraction of the smoothing radius. Safety net only.")]
    public float maxCorrectionRatio = 0.6f;
    [Tooltip("Softening of the PBF denominator, relative to the rest state (scale-independent). Higher = softer + more stable.")]
    public float relaxation = 1f;
    [Tooltip("Scales type.viscosity into the fraction of relative velocity removed per step.")]
    public float viscosityScale = 0.5f;

    [Header("Burst / Jobs")]
    [Tooltip("Run the passes on worker threads. Below Parallel Threshold they run on the main thread (still Burst-compiled).")]
    public bool useJobs = true;
    [Tooltip("Scheduling a job costs a few microseconds; below this many particles one thread is faster.")]
    public int parallelThreshold = 400;
    [Range(8, 256)] public int jobBatchSize = 32;

    [Header("Types")]
    public List<FluidType> types = new List<FluidType>()
    {
        new FluidType() { name = "Water", restDensity = 20f, restDensityScale = 1f,    stiffness = 0.8f, viscosity = 0.2f, gravityMultiplier = 1f },
        new FluidType() { name = "Oil",   restDensity = 15f, restDensityScale = 0.75f, stiffness = 0.6f, viscosity = 0.6f, gravityMultiplier = 1f },
    };
    public int defaultType = 0;

    [Header("Source (drag PhysicsPose objects, never Finger sphere)")]
    public List<PhysicsPose> manualPoses = new List<PhysicsPose>();
    public List<PhysicsPose> autoSyncSource;
    public bool autoSyncEveryFixedUpdate = false;

    [Header("Calibration")]
    [Tooltip("ON: rest density = (mean body mass) x (kernel sum of a square lattice at Reference Spacing). " +
             "Can never drift out of sync with the real bodies. OFF: use FluidType.restDensity as typed.")]
    public bool autoCalibrate = true;
    [Tooltip("Particle spacing at rest. The spawner sets this for you.")]
    public float referenceSpacing = 0.4f;
    [Tooltip("Only used by the 'Calibrate' context menu / when Auto Calibrate is OFF.")]
    public float referenceMass = 1.28f;

    [Header("Debug")]
    public bool showParticlesViaPhysicsDraw = true;
    public bool showLogs = false;

    [Header("Diagnostics (written by the sim, read-only)")]
    public bool updateDiagnostics = true;
    [Tooltip("Mean rho/rho0 at the start of the step (sampled every 8th step). Stuck well below 1 = nothing is compressed = no pressure.")]
    public float diagAvgDensityRatio;
    [Tooltip("Particles with rho > rho0 (the ones the pressure solver is acting on).")]
    public int diagCompressed;
    [Tooltip("Particles currently outside the static grid (clamped into border cells).")]
    public int diagOutsideGrid;
    [Tooltip("Cost of the whole fluid pass on the main thread, milliseconds.")]
    public float diagStepMs;

    public int ParticleCount => count;

    // ------------------------------------------------------------------ registry
    PhysicsBody[] bodies;
    float[] baseGs;      // the body's own gravityScale when it was registered
    float[] appliedGs;   // what we last wrote to body.gravityScale (NaN = not written yet)
    PhysicsBody.BatchForce[] forceBatch;
    NativeArray<float> nMass;   // job-visible registry data
    NativeArray<int> nType;
    int count;
    float massSum;
    readonly Dictionary<PhysicsBody, int> indexOf = new Dictionary<PhysicsBody, int>();
    readonly HashSet<PhysicsPose> simPoses = new HashSet<PhysicsPose>();

    // ------------------------------------------------------------------ per-step native scratch
    NativeArray<float2> nX, nV, nVPred, nPos0, nPredA, nPredB, nVCon, nForce;
    NativeArray<float> nGs, nLambda, nRatio;
    NativeArray<byte> nActive;
    NativeArray<int> nCellOf, nCellItems, nCellStart, nCellFill;
    NativeArray<PbfTypeParams> nTypes;
    float[] typeGm;

    // lattice-sum cache for calibration
    float cacheH = -1f, cacheSp = -1f, cacheSum;

    static readonly FluidType FallbackType = new FluidType();
    readonly System.Diagnostics.Stopwatch stepWatch = new System.Diagnostics.Stopwatch();
    int stepCounter;
    float gridMinX, gridMinY, gridMaxX, gridMaxY;

    static bool Finite(float f) { return !(float.IsNaN(f) || float.IsInfinity(f)); }
    static bool Finite(Vector2 v) { return Finite(v.x) && Finite(v.y); }
    static float ReadMass(PhysicsBody b) { float m = b.mass; return (m > 0f && Finite(m)) ? m : 1f; }

    // ================================================================== public API
    public void AddParticle(PhysicsBody body, int typeId)
    {
        if (!body.isValid) return;
        int t = ClampType(typeId);
        // Already registered: only the type changes. (Re-reading gravityScale here would capture OUR value.)
        if (indexOf.TryGetValue(body, out int existing)) { nType[existing] = t; return; }
        EnsureCapacity(count + 1);
        int i = count++;
        float gs = body.gravityScale;
        float m = ReadMass(body);
        bodies[i] = body;
        baseGs[i] = Finite(gs) ? gs : 1f;
        appliedGs[i] = float.NaN;
        nMass[i] = m;
        nType[i] = t;
        massSum += m;
        indexOf[body] = i;
    }
    public void AddParticle(PhysicsBody body) { AddParticle(body, ClampType(defaultType)); }

    public void AddPose(PhysicsPose pose, int typeId)
    {
        if (pose == null) return;
        simPoses.Add(pose);
        if (!manualPoses.Contains(pose)) manualPoses.Add(pose);
        try
        {
            PhysicsBody b = pose.body;
            if (b.isValid) AddParticle(b, typeId);
        }
        catch { }
    }

    public void SetParticleType(PhysicsBody body, int typeId)
    {
        if (!body.isValid) return;
        int t = ClampType(typeId);
        if (indexOf.TryGetValue(body, out int i)) nType[i] = t; else AddParticle(body, t);
    }

    public void RemoveParticle(PhysicsBody body)
    {
        if (indexOf.TryGetValue(body, out int i)) RemoveAt(i, true);
    }

    public void RemovePose(PhysicsPose pose)
    {
        if (pose == null) return;
        try
        {
            PhysicsBody b = pose.body;
            if (b.isValid && simPoses.Contains(pose)) RemoveParticle(b);
        }
        catch { }
        simPoses.Remove(pose); manualPoses.Remove(pose);
        if (autoSyncSource != null) autoSyncSource.Remove(pose);
    }

    public void ClearParticles()
    {
        RestoreGravityScales();
        indexOf.Clear(); simPoses.Clear(); manualPoses.Clear();
        for (int i = 0; i < count; i++) bodies[i] = default;
        count = 0; massSum = 0f;
    }
    public void Clear() => ClearParticles();

    public int GetParticleType(PhysicsBody body)
    {
        return indexOf.TryGetValue(body, out int i) ? nType[i] : ClampType(defaultType);
    }

    public int ClampType(int t)
    {
        if (types == null || types.Count == 0) return 0;
        if (t < 0) return 0;
        if (t >= types.Count) return types.Count - 1;
        return t;
    }

    /// <summary>Re-read body masses (call if you change masses after registering).</summary>
    public void RefreshParticleMasses()
    {
        massSum = 0f;
        for (int i = 0; i < count; i++)
        {
            if (bodies[i].isValid) nMass[i] = ReadMass(bodies[i]);
            massSum += nMass[i];
        }
    }

    /// <summary>Called by the spawner so smoothing radius + calibration always match what was actually spawned.</summary>
    public void ConfigureReference(float spacing, float particleMass)
    {
        referenceSpacing = Mathf.Max(0.01f, spacing);
        if (particleMass > 1e-6f) referenceMass = particleMass;
        ApplyCalibrationToTypes();
    }

    [ContextMenu("Sync From List")]
    public void SyncFromList()
    {
        simPoses.Clear();
        int def = ClampType(defaultType);
        AddPoseBodies(manualPoses, def);
        AddPoseBodies(autoSyncSource, def);
        CleanPoses(manualPoses);
        PruneInvalid();
        if (showLogs) Debug.Log($"[PbfFluidLite] SyncFromList: {count} particles.");
    }

    [ContextMenu("Calibrate Rest Density From Spacing")]
    public void CalibrateRestDensity()
    {
        float rho0 = ApplyCalibrationToTypes();
        Debug.Log($"[PbfFluidLite] rest density = {rho0:F3} (h={EffectiveSmoothing}, spacing={Mathf.Max(0.01f, referenceSpacing)}, mass={referenceMass}).");
    }

    // ================================================================== lifecycle
    void OnEnable() { SyncFromList(); ApplyDrawOptions(); }

    void OnDisable() { RestoreGravityScales(); }

    void OnDestroy() { DisposeNative(); }

    void OnValidate()
    {
        if (smoothingRadius < 0.05f) smoothingRadius = 0.05f;
        if (smoothingToSpacing < 1.2f) smoothingToSpacing = 1.2f;
        if (gridColumns < 1) gridColumns = 1;
        if (gridRows < 1) gridRows = 1;
        if (gridColumns > 1024) gridColumns = 1024;
        if (gridRows > 1024) gridRows = 1024;
        if (iterations < 1) iterations = 1;
        if (iterations > 10) iterations = 10;
        if (maxSpeed < 0.1f) maxSpeed = 0.1f;
        if (maxCorrectionRatio < 0.05f) maxCorrectionRatio = 0.05f;
        if (relaxation < 1e-6f) relaxation = 1e-6f;
        if (viscosityScale < 0f) viscosityScale = 0f;
        if (parallelThreshold < 0) parallelThreshold = 0;
        if (referenceSpacing < 0.01f) referenceSpacing = 0.01f;
        if (referenceMass <= 0f) referenceMass = 1f;
        if (types == null) types = new List<FluidType>();
        if (types.Count == 0) types.Add(new FluidType());
        defaultType = ClampType(defaultType);
        ApplyDrawOptions();
    }

    void FixedUpdate()
    {
        if (!runInFixedUpdate || !enablePBF) return;
        float dt = Time.fixedDeltaTime;
        if (dt <= 0f) return;
        if (autoSyncEveryFixedUpdate) SyncFromList();
        else if (count == 0 && manualPoses.Count > 0) SyncFromList();
        StepPBF(dt);
    }

    // ================================================================== the fluid step
    void StepPBF(float dt)
    {
        PruneInvalid();
        int n = count;
        if (n == 0 || types == null || types.Count == 0) return;
        stepWatch.Restart();
        stepCounter++;
        bool doDiag = updateDiagnostics && (stepCounter & 7) == 0;

        int cols = Mathf.Max(1, gridColumns), rows = Mathf.Max(1, gridRows);
        EnsureGrid(cols * rows);

        float h = EffectiveSmoothing;
        float h2 = h * h;
        float h6 = h2 * h2 * h2;
        float poly6 = 315f / (64f * Mathf.PI * h6 * h2 * h);   // 315 / (64 pi h^9)
        float spikyGrad = -45f / (Mathf.PI * h6);
        float selfW = poly6 * h6;
        float invDt = 1f / dt;
        float eps = relaxation * 0.25f / h2;                    // 0.25/h^2 keeps the softening scale-independent
        float maxCorr = maxCorrectionRatio * h;
        Vector2 gravity = bodies[0].world.gravity;              // the world the particles actually live in

        gridMinX = gridOrigin.x; gridMinY = gridOrigin.y;
        gridMaxX = gridMinX + cols * h; gridMaxY = gridMinY + rows * h;

        // ---- per-type table (tiny) ----
        int typeCount = types.Count;
        EnsureTypeTable(typeCount);
        float baseRho0 = autoCalibrate ? (massSum / n) * CurrentLatticeSum(h) : 0f;
        for (int t = 0; t < typeCount; t++)
        {
            FluidType ft = types[t] ?? FallbackType;
            nTypes[t] = new PbfTypeParams
            {
                rho0 = autoCalibrate
                    ? Mathf.Max(0.001f, baseRho0 * Mathf.Max(0.05f, ft.restDensityScale))
                    : Mathf.Max(0.001f, ft.restDensity),
                stiffness = Mathf.Clamp01(ft.stiffness),
                viscosity = Mathf.Clamp01(ft.viscosity * viscosityScale),
            };
            typeGm[t] = ft.gravityMultiplier;
        }

        // ---- 1. read engine state (two batched native calls), push gravity scales ----
        int outside = 0;
#if PBF_PER_BODY_READ
        for (int i = 0; i < n; i++)
            if (StoreState(i, bodies[i].position, bodies[i].linearVelocity, typeCount)) outside++;
#else
        var span = new System.ReadOnlySpan<PhysicsBody>(bodies, 0, n);
        NativeArray<PhysicsBody.BatchTransform> tr = PhysicsBody.GetBatchTransform(span, Allocator.Temp);
        NativeArray<PhysicsBody.BatchVelocity> vl = PhysicsBody.GetBatchVelocity(span, Allocator.Temp);
        for (int i = 0; i < n; i++)
        {
            Vector2 px = tr[i].position;
            Vector2 pv = vl[i].linearVelocity;
            if (StoreState(i, px, pv, typeCount)) outside++;
        }
        tr.Dispose();
        vl.Dispose();
#endif

        // ---- 2. Burst jobs ----
        bool par = useJobs && n >= parallelThreshold;
        int bs = Mathf.Max(1, jobBatchSize);
        JobHandle hnd = default;

        var predict = new PbfPredictJob
        {
            x = nX, v = nV, gs = nGs, gravity = gravity, dt = dt,
            vPred = nVPred, pos0 = nPos0, pred = nPredA,
        };
        if (par) hnd = predict.Schedule(n, bs, hnd); else predict.Run(n);

        NativeArray<float2> cur = nPredA, nxt = nPredB;
        var grid = new PbfGridJob
        {
            p = cur, active = nActive, cellOf = nCellOf, cellStart = nCellStart, cellFill = nCellFill, cellItems = nCellItems,
            origin = gridOrigin, invCell = 1f / h, cols = cols, rows = rows, n = n,
        };
        if (par) hnd = grid.Schedule(hnd); else grid.Run();

        for (int it = 0; it < iterations; it++)
        {
            var dens = new PbfDensityJob
            {
                pred = cur, mass = nMass, type = nType, types = nTypes,
                cellOf = nCellOf, cellStart = nCellStart, cellItems = nCellItems,
                cols = cols, rows = rows, typeCount = typeCount,
                h = h, h2 = h2, poly6 = poly6, spikyGrad = spikyGrad, selfW = selfW, eps = eps,
                writeRatio = doDiag && it == 0, lambda = nLambda, ratioOut = nRatio,
            };
            if (par) hnd = dens.Schedule(n, bs, hnd); else dens.Run(n);

            var corr = new PbfCorrectionJob
            {
                pred = cur, lambda = nLambda, mass = nMass, type = nType, types = nTypes,
                cellOf = nCellOf, cellStart = nCellStart, cellItems = nCellItems,
                cols = cols, rows = rows, typeCount = typeCount,
                h = h, h2 = h2, spikyGrad = spikyGrad, maxCorr = maxCorr, predNext = nxt,
            };
            if (par) hnd = corr.Schedule(n, bs, hnd); else corr.Run(n);

            NativeArray<float2> tmp = cur; cur = nxt; nxt = tmp;   // double buffer
            grid.p = cur;                                          // re-bucket the moved particles
            if (par) hnd = grid.Schedule(hnd); else grid.Run();
        }

        var vel = new PbfVelocityJob
        {
            vPred = nVPred, pred = cur, pos0 = nPos0, invDt = invDt, maxSpeed = maxSpeed, vCon = nVCon,
        };
        if (par) hnd = vel.Schedule(n, bs, hnd); else vel.Run(n);

        var visc = new PbfViscosityForceJob
        {
            vCon = nVCon, vPred = nVPred, pred = cur, mass = nMass, type = nType, types = nTypes,
            cellOf = nCellOf, cellStart = nCellStart, cellItems = nCellItems,
            cols = cols, rows = rows, typeCount = typeCount, h2 = h2, invDt = invDt, force = nForce,
        };
        if (par) hnd = visc.Schedule(n, bs, hnd); else visc.Run(n);

        if (par) hnd.Complete();

        // ---- 3. hand the result to the engine as ONE batched force.
        // The engine adds gravity itself; the jobs already expressed the force relative to the gravity-only prediction.
        int nf = 0;
        for (int i = 0; i < n; i++)
        {
            float2 f2 = nForce[i];
            if (f2.x * f2.x + f2.y * f2.y < 1e-10f) continue;
            Vector2 f = f2;
            if (!Finite(f)) continue;
            PhysicsBody.BatchForce bf = new PhysicsBody.BatchForce(bodies[i]);
            bf.ApplyForceToCenter(f);
            forceBatch[nf++] = bf;
        }
        if (nf > 0) PhysicsBody.SetBatchForce(new System.ReadOnlySpan<PhysicsBody.BatchForce>(forceBatch, 0, nf));

        if (doDiag)
        {
            float sum = 0f; int comp = 0, act = 0;
            for (int i = 0; i < n; i++)
            {
                if (nActive[i] == 0) continue;
                float r = nRatio[i]; sum += r; act++;
                if (r > 1f) comp++;
            }
            diagAvgDensityRatio = act > 0 ? sum / act : 0f;
            diagCompressed = comp;
            diagOutsideGrid = outside;
        }
        diagStepMs = (float)stepWatch.Elapsed.TotalMilliseconds;
        if (showLogs && (stepCounter % 100) == 0)
            Debug.Log($"[PbfFluidLite] n={n} {(par ? "jobs" : "main-thread burst")} avg rho/rho0={diagAvgDensityRatio:F2} " +
                      $"compressed={diagCompressed} outsideGrid={diagOutsideGrid} cost={diagStepMs:F2} ms");
    }

    /// <summary>Copies one particle's engine state into the job arrays and pushes the gravity scale. Returns true if outside the grid.</summary>
    bool StoreState(int i, Vector2 x, Vector2 v, int typeCount)
    {
        int t = nType[i]; if (t >= typeCount) t = typeCount - 1;
        float gsEff = baseGs[i] * typeGm[t];
        if (appliedGs[i] != gsEff) { bodies[i].gravityScale = gsEff; appliedGs[i] = gsEff; }   // the ENGINE applies gravity
        nGs[i] = gsEff;
        if (!Finite(x) || !Finite(v))
        {
            nActive[i] = 0; nX[i] = new float2(0f, 0f); nV[i] = new float2(0f, 0f);
            return false;
        }
        nActive[i] = 1; nX[i] = x; nV[i] = v;
        return x.x < gridMinX || x.x >= gridMaxX || x.y < gridMinY || x.y >= gridMaxY;
    }

    // ================================================================== bookkeeping
    void EnsureCapacity(int need)
    {
        if (bodies != null && need <= bodies.Length) return;
        int cap = Mathf.Max(64, need, bodies == null ? 0 : bodies.Length * 2);
        System.Array.Resize(ref bodies, cap);
        System.Array.Resize(ref baseGs, cap);
        System.Array.Resize(ref appliedGs, cap);
        System.Array.Resize(ref forceBatch, cap);
        Realloc(ref nMass, cap, count);        // registry data: keep contents
        Realloc(ref nType, cap, count);
        Realloc(ref nX, cap, 0);               // scratch: contents rewritten every step
        Realloc(ref nV, cap, 0);
        Realloc(ref nVPred, cap, 0);
        Realloc(ref nPos0, cap, 0);
        Realloc(ref nPredA, cap, 0);
        Realloc(ref nPredB, cap, 0);
        Realloc(ref nVCon, cap, 0);
        Realloc(ref nForce, cap, 0);
        Realloc(ref nGs, cap, 0);
        Realloc(ref nLambda, cap, 0);
        Realloc(ref nRatio, cap, 0);
        Realloc(ref nActive, cap, 0);
        Realloc(ref nCellOf, cap, 0);
        Realloc(ref nCellItems, cap, 0);
    }

    static void Realloc<T>(ref NativeArray<T> a, int cap, int keep) where T : struct
    {
        var n = new NativeArray<T>(cap, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        if (a.IsCreated)
        {
            if (keep > 0) NativeArray<T>.Copy(a, n, keep);
            a.Dispose();
        }
        a = n;
    }

    void EnsureGrid(int total)
    {
        if (nCellStart.IsCreated && nCellStart.Length == total + 1) return;
        if (nCellStart.IsCreated) nCellStart.Dispose();
        if (nCellFill.IsCreated) nCellFill.Dispose();
        nCellStart = new NativeArray<int>(total + 1, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        nCellFill = new NativeArray<int>(total + 1, Allocator.Persistent, NativeArrayOptions.ClearMemory);
    }

    void EnsureTypeTable(int need)
    {
        if (nTypes.IsCreated && nTypes.Length >= need) return;
        if (nTypes.IsCreated) nTypes.Dispose();
        int cap = Mathf.Max(8, need);
        nTypes = new NativeArray<PbfTypeParams>(cap, Allocator.Persistent, NativeArrayOptions.ClearMemory);
        typeGm = new float[cap];
    }

    void DisposeNative()
    {
        Dispose(ref nMass); Dispose(ref nType);
        Dispose(ref nX); Dispose(ref nV); Dispose(ref nVPred); Dispose(ref nPos0);
        Dispose(ref nPredA); Dispose(ref nPredB); Dispose(ref nVCon); Dispose(ref nForce);
        Dispose(ref nGs); Dispose(ref nLambda); Dispose(ref nRatio); Dispose(ref nActive);
        Dispose(ref nCellOf); Dispose(ref nCellItems); Dispose(ref nCellStart); Dispose(ref nCellFill);
        Dispose(ref nTypes);
    }
    static void Dispose<T>(ref NativeArray<T> a) where T : struct { if (a.IsCreated) a.Dispose(); }

    void RemoveAt(int i, bool restoreGravity)
    {
        PhysicsBody b = bodies[i];
        if (restoreGravity && b.isValid) b.gravityScale = baseGs[i];
        indexOf.Remove(b);
        massSum -= nMass[i];
        int last = count - 1;
        if (i != last)
        {
            bodies[i] = bodies[last];
            baseGs[i] = baseGs[last]; appliedGs[i] = appliedGs[last];
            nMass[i] = nMass[last]; nType[i] = nType[last];
            indexOf[bodies[i]] = i;
        }
        bodies[last] = default;
        count--;
        if (count == 0) massSum = 0f;
    }

    void PruneInvalid()
    {
        for (int i = count - 1; i >= 0; i--)
            if (!bodies[i].isValid) RemoveAt(i, false);
    }

    void RestoreGravityScales()
    {
        for (int i = 0; i < count; i++)
        {
            if (bodies[i].isValid) bodies[i].gravityScale = baseGs[i];
            appliedGs[i] = float.NaN;
        }
    }

    void AddPoseBodies(List<PhysicsPose> list, int def)
    {
        if (list == null) return;
        for (int i = 0; i < list.Count; i++)
        {
            PhysicsPose p = list[i];
            if (p == null) continue;
            simPoses.Add(p);
            try
            {
                PhysicsBody b = p.body;
                if (b.isValid && !indexOf.ContainsKey(b)) AddParticle(b, def);
            }
            catch { }
        }
    }

    void CleanPoses(List<PhysicsPose> list)
    {
        if (list == null) return;
        for (int i = list.Count - 1; i >= 0; i--) if (list[i] == null) list.RemoveAt(i);
    }

    void ApplyDrawOptions()
    {
        try
        {
            PhysicsWorld w = PhysicsWorld.defaultWorld;
            w.drawOptions = showParticlesViaPhysicsDraw ? PhysicsWorld.DrawOptions.AllShapes : PhysicsWorld.DrawOptions.Off;
        }
        catch (System.Exception ex) { if (showLogs) Debug.LogWarning("[PbfFluidLite] drawOptions: " + ex.Message); }
    }

    // ================================================================== calibration
    // Sum of the poly6 kernel over a square lattice (self included) with the given spacing.
    // Iterates integer lattice indices so the origin is ALWAYS a lattice point.
    float CurrentLatticeSum(float h)
    {
        float sp = Mathf.Max(0.01f, referenceSpacing);
        if (h != cacheH || sp != cacheSp)
        {
            float h2 = h * h;
            float poly6 = 315f / (64f * Mathf.PI * Mathf.Pow(h, 9));
            int K = Mathf.CeilToInt(h / sp);
            float sum = 0f;
            for (int j = -K; j <= K; j++)
                for (int i = -K; i <= K; i++)
                {
                    float x = i * sp, y = j * sp;
                    float r2 = x * x + y * y;
                    if (r2 >= h2) continue;
                    float q = h2 - r2; sum += poly6 * q * q * q;
                }
            cacheH = h; cacheSp = sp; cacheSum = sum;
        }
        return cacheSum;
    }

    float ApplyCalibrationToTypes()
    {
        float rho0 = Mathf.Max(0.001f, referenceMass * CurrentLatticeSum(EffectiveSmoothing));
        if (types == null || types.Count == 0) types = new List<FluidType>() { new FluidType() };
        foreach (var ft in types) if (ft != null) ft.restDensity = rho0 * ft.restDensityScale;
        return rho0;
    }

    // ================================================================== gizmos
    void DrawGridLines()
    {
        int cols = Mathf.Max(1, gridColumns), rows = Mathf.Max(1, gridRows);
        float cs = EffectiveSmoothing;
        Vector2 min = gridOrigin;
        Vector2 max = gridOrigin + new Vector2(cols * cs, rows * cs);
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.9f);
        Gizmos.DrawLine(new Vector3(min.x, min.y, 0f), new Vector3(max.x, min.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, min.y, 0f), new Vector3(max.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, max.y, 0f), new Vector3(min.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(min.x, max.y, 0f), new Vector3(min.x, min.y, 0f));
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.35f);
        for (int i = 1; i < cols && i < 200; i++)
        { float x = min.x + i * cs; Gizmos.DrawLine(new Vector3(x, min.y, 0f), new Vector3(x, max.y, 0f)); }
        for (int i = 1; i < rows && i < 200; i++)
        { float y = min.y + i * cs; Gizmos.DrawLine(new Vector3(min.x, y, 0f), new Vector3(max.x, y, 0f)); }
    }
    void OnDrawGizmos() { if (!showGridGizmo || !drawGridWhenUnselected) return; DrawGridLines(); }
    void OnDrawGizmosSelected() { if (!showGridGizmo) return; DrawGridLines(); }
}
