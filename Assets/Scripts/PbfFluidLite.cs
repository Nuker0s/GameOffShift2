using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// PBF-lite fluid layered on top of Physics Core 2D.
///
/// WHO DOES WHAT
///   Physics Core 2D (engine): integration, gravity (body.gravityScale), damping, mass, shape contacts,
///                             walls / other bodies / Finger, sleeping, sub-stepping.
///   This script:              the one thing the engine has no concept of: the inter-particle density
///                             constraint (pressure) and viscosity.
///
/// HOW
///   FixedUpdate runs immediately before the engine step. Each step we READ position + linearVelocity,
///   predict where the engine is about to take every particle, run PBF on the predicted positions, and
///   hand the result back as ONE batched force (F = m * dv / dt) through PhysicsBody.SetBatchForce.
///   We never write position or velocity, so there is exactly one integrator: the engine.
///
/// ASSUMPTION
///   The default world simulates in FixedUpdate (the default), i.e. one engine step per FixedUpdate and
///   user forces are cleared after each step.
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

    [Header("Neighbour search (grid auto-fits the particles every step)")]
    public float cellSize = 0.5f;
    public float smoothingRadius = 0.5f;

    [Header("Solver")]
    [Range(1, 10)] public int iterations = 3;
    public bool enablePBF = true;
    [Tooltip("Untick to suspend the fluid pass. The engine keeps simulating the bodies as plain circles.")]
    public bool runInFixedUpdate = true;
    [Tooltip("Cap on how fast the fluid pass may ACCELERATE a particle (m/s).")]
    public float maxSpeed = 20f;
    [Tooltip("Cap on the positional correction per iteration (m). Safety net only.")]
    public float maxCorrection = 0.3f;
    [Tooltip("Softening added to the PBF denominator. Higher = softer + more stable.")]
    public float relaxation = 1f;
    [Tooltip("Scales type.viscosity into the fraction of relative velocity removed per step.")]
    public float viscosityScale = 0.5f;

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
    public float referenceSpacing = 0.25f;
    [Tooltip("Only used by the 'Calibrate' context menu / when Auto Calibrate is OFF.")]
    public float referenceMass = 0.5f;

    [Header("Debug")]
    public bool showParticlesViaPhysicsDraw = true;
    public bool showGridGizmo = true;
    public bool drawGridWhenUnselected = true;
    public bool showLogs = false;

    [Header("Diagnostics (written by the sim, read-only)")]
    [Tooltip("Mean rho/rho0 at the start of the step. Stuck well below 1 = nothing is ever compressed = no pressure.")]
    public float diagAvgDensityRatio;
    [Tooltip("Particles with rho > rho0 (the ones the pressure solver is acting on).")]
    public int diagCompressed;
    [Tooltip("Cost of the whole fluid pass, milliseconds.")]
    public float diagStepMs;

    public int ParticleCount => count;

    // ------------------------------------------------------------------ registry (struct-of-arrays)
    PhysicsBody[] bodies;
    int[] typeOf;
    float[] baseGs;      // the body's own gravityScale when it was registered
    float[] appliedGs;   // what we last wrote to body.gravityScale (NaN = not written yet)
    float[] mass;
    int count;
    readonly Dictionary<PhysicsBody, int> indexOf = new Dictionary<PhysicsBody, int>();
    readonly HashSet<PhysicsPose> simPoses = new HashSet<PhysicsPose>();

    // ------------------------------------------------------------------ per-step scratch (never allocated in the loop)
    Vector2[] pred, pos0, vPred, vCon, vOut, delta;
    float[] lambda, rho0Arr, stiffArr, viscArr;
    int[] cellOf, cellItems;
    PhysicsBody.BatchForce[] forceBatch;

    // spatial grid, rebuilt from the particle bounding box so particles can never "leave" it
    const int MaxCells = 1 << 16;
    int[] cellStart, cellFill;
    int cellsX, cellsY;
    float effectiveCell = 0.5f, invCell = 2f;
    Vector2 gridMin;

    // lattice-sum cache for calibration
    float cacheH = -1f, cacheSp = -1f, cacheSum;

    readonly System.Diagnostics.Stopwatch stepWatch = new System.Diagnostics.Stopwatch();
    int stepCounter;

    static bool Finite(float f) { return !(float.IsNaN(f) || float.IsInfinity(f)); }
    static bool Finite(Vector2 v) { return Finite(v.x) && Finite(v.y); }
    static float ReadMass(PhysicsBody b) { float m = b.mass; return (m > 0f && Finite(m)) ? m : 1f; }

    // ================================================================== public API
    public void AddParticle(PhysicsBody body, int typeId)
    {
        if (!body.isValid) return;
        int t = ClampType(typeId);
        // Already registered: only the type changes. (Re-reading gravityScale here would capture OUR value.)
        if (indexOf.TryGetValue(body, out int existing)) { typeOf[existing] = t; return; }
        EnsureCapacity(count + 1);
        int i = count++;
        float gs = body.gravityScale;
        bodies[i] = body;
        typeOf[i] = t;
        baseGs[i] = Finite(gs) ? gs : 1f;
        appliedGs[i] = float.NaN;
        mass[i] = ReadMass(body);
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
        if (indexOf.TryGetValue(body, out int i)) typeOf[i] = t; else AddParticle(body, t);
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
        count = 0;
    }
    public void Clear() => ClearParticles();

    public int GetParticleType(PhysicsBody body)
    {
        return indexOf.TryGetValue(body, out int i) ? typeOf[i] : ClampType(defaultType);
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
        for (int i = 0; i < count; i++) if (bodies[i].isValid) mass[i] = ReadMass(bodies[i]);
    }

    /// <summary>Called by the spawner so calibration always matches what was actually spawned.</summary>
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
        Debug.Log($"[PbfFluidLite] rest density = {rho0:F3} (h={Mathf.Max(0.05f, smoothingRadius)}, spacing={Mathf.Max(0.01f, referenceSpacing)}, mass={referenceMass}).");
    }

    // ================================================================== lifecycle
    void OnEnable() { SyncFromList(); ApplyDrawOptions(); }

    void OnDisable() { RestoreGravityScales(); }

    void OnValidate()
    {
        if (smoothingRadius < 0.05f) smoothingRadius = 0.05f;
        if (cellSize < 0.05f) cellSize = 0.05f;
        if (iterations < 1) iterations = 1;
        if (iterations > 10) iterations = 10;
        if (maxSpeed < 0.1f) maxSpeed = 0.1f;
        if (maxCorrection < 0.01f) maxCorrection = 0.01f;
        if (relaxation < 1e-6f) relaxation = 1e-6f;
        if (viscosityScale < 0f) viscosityScale = 0f;
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

        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h;
        float h6 = h2 * h2 * h2;
        float poly6 = 315f / (64f * Mathf.PI * h6 * h2 * h);   // 315 / (64 pi h^9)
        float spikyGrad = -45f / (Mathf.PI * h6);
        float selfW = poly6 * h6;
        float invDt = 1f / dt;
        Vector2 gravity = bodies[0].world.gravity;              // the world the particles actually live in

        float massSum = 0f;
        for (int i = 0; i < n; i++) massSum += mass[i];
        float baseRho0 = autoCalibrate ? (massSum / n) * CurrentLatticeSum(h) : 0f;

        // ---- 1. read engine state, predict where the engine is about to put each particle
        for (int i = 0; i < n; i++)
        {
            FluidType ft = types[ClampType(typeOf[i])];
            float gsEff = baseGs[i] * ft.gravityMultiplier;
            if (appliedGs[i] != gsEff) { bodies[i].gravityScale = gsEff; appliedGs[i] = gsEff; }   // engine applies gravity

            Vector2 x = bodies[i].position;
            Vector2 v = bodies[i].linearVelocity;
            Vector2 vP = v + gravity * (gsEff * dt);
            Vector2 p0 = x + vP * dt;
            vPred[i] = vP; pos0[i] = p0; pred[i] = p0;

            rho0Arr[i] = autoCalibrate
                ? Mathf.Max(0.001f, baseRho0 * Mathf.Max(0.05f, ft.restDensityScale))
                : Mathf.Max(0.001f, ft.restDensity);
            stiffArr[i] = Mathf.Clamp01(ft.stiffness);
            viscArr[i] = Mathf.Clamp01(ft.viscosity * viscosityScale);
        }

        // ---- 2. PBF density constraint on the predicted positions
        BuildGrid(pred, n);
        float sumRatio = 0f; int compressed = 0;
        for (int it = 0; it < iterations; it++)
        {
            // pass A: density + lambda
            for (int i = 0; i < n; i++)
            {
                int c0 = cellOf[i];
                if (c0 < 0) { lambda[i] = 0f; continue; }
                int cy0 = c0 / cellsX, cx0 = c0 - cy0 * cellsX;
                Vector2 pi = pred[i];
                float invRho0 = 1f / rho0Arr[i];
                float dens = mass[i] * selfW;
                Vector2 gI = Vector2.zero; float s2 = 0f;
                for (int oy = -1; oy <= 1; oy++)
                {
                    int ny = cy0 + oy; if (ny < 0 || ny >= cellsY) continue;
                    for (int ox = -1; ox <= 1; ox++)
                    {
                        int nx = cx0 + ox; if (nx < 0 || nx >= cellsX) continue;
                        int c = ny * cellsX + nx;
                        int end = cellStart[c + 1];
                        for (int k = cellStart[c]; k < end; k++)
                        {
                            int j = cellItems[k]; if (j == i) continue;
                            Vector2 d = pi - pred[j];
                            float r2 = d.x * d.x + d.y * d.y;
                            if (r2 >= h2 || r2 < 1e-12f) continue;
                            float q = h2 - r2;
                            float mj = mass[j];
                            dens += mj * poly6 * q * q * q;
                            float r = Mathf.Sqrt(r2);
                            float hr = h - r;
                            // dC_i/dp_i contribution of neighbour j: (m_j / rho0) * gradW
                            Vector2 g = d * (spikyGrad * hr * hr / r * mj * invRho0);
                            gI += g; s2 += g.x * g.x + g.y * g.y;
                        }
                    }
                }
                float ratio = dens * invRho0;
                if (it == 0) { sumRatio += ratio; if (ratio > 1f) compressed++; }
                float C = ratio - 1f;
                if (C <= 0f) { lambda[i] = 0f; continue; }      // only resist compression
                s2 += gI.x * gI.x + gI.y * gI.y;
                lambda[i] = -C / (s2 + relaxation);
            }

            // pass B: position correction.  dp_i = (1/rho0) * sum_j (m_j*lambda_i + m_i*lambda_j) * gradW_ij
            for (int i = 0; i < n; i++)
            {
                int c0 = cellOf[i];
                if (c0 < 0) { delta[i] = Vector2.zero; continue; }
                int cy0 = c0 / cellsX, cx0 = c0 - cy0 * cellsX;
                Vector2 pi = pred[i];
                float li = lambda[i], mi = mass[i];
                Vector2 corr = Vector2.zero;
                for (int oy = -1; oy <= 1; oy++)
                {
                    int ny = cy0 + oy; if (ny < 0 || ny >= cellsY) continue;
                    for (int ox = -1; ox <= 1; ox++)
                    {
                        int nx = cx0 + ox; if (nx < 0 || nx >= cellsX) continue;
                        int c = ny * cellsX + nx;
                        int end = cellStart[c + 1];
                        for (int k = cellStart[c]; k < end; k++)
                        {
                            int j = cellItems[k]; if (j == i) continue;
                            float lj = lambda[j];
                            if (li == 0f && lj == 0f) continue;
                            Vector2 d = pi - pred[j];
                            float r2 = d.x * d.x + d.y * d.y;
                            if (r2 >= h2 || r2 < 1e-12f) continue;
                            float r = Mathf.Sqrt(r2);
                            float hr = h - r;
                            corr += d * (spikyGrad * hr * hr / r * (mass[j] * li + mi * lj));
                        }
                    }
                }
                corr *= stiffArr[i] / rho0Arr[i];
                float mag = corr.magnitude;
                if (mag > maxCorrection) corr *= maxCorrection / mag;
                delta[i] = corr;
            }

            for (int i = 0; i < n; i++) pred[i] += delta[i];
            BuildGrid(pred, n);   // fresh grid for the next iteration and for the viscosity pass
        }

        // ---- 3. velocity the constraint wants (PBF: v = (x* - x)/dt, expressed relative to the engine's own step)
        for (int i = 0; i < n; i++)
        {
            Vector2 vP = vPred[i];
            Vector2 vC = vP + (pred[i] - pos0[i]) * invDt;
            float cap = Mathf.Max(maxSpeed, vP.magnitude);       // limit how much we ADD, never brake an existing speed
            float sp = vC.magnitude;
            if (sp > cap && sp > 1e-6f) vC *= cap / sp;
            vCon[i] = vC;
        }

        // ---- 4. XSPH viscosity: blend toward the kernel-weighted mean velocity of the neighbours
        for (int i = 0; i < n; i++)
        {
            Vector2 vc = vCon[i];
            float visc = viscArr[i];
            int c0 = cellOf[i];
            if (visc <= 0f || c0 < 0) { vOut[i] = vc; continue; }
            int cy0 = c0 / cellsX, cx0 = c0 - cy0 * cellsX;
            Vector2 pi = pred[i];
            Vector2 acc = Vector2.zero; float wSum = 0f;
            for (int oy = -1; oy <= 1; oy++)
            {
                int ny = cy0 + oy; if (ny < 0 || ny >= cellsY) continue;
                for (int ox = -1; ox <= 1; ox++)
                {
                    int nx = cx0 + ox; if (nx < 0 || nx >= cellsX) continue;
                    int c = ny * cellsX + nx;
                    int end = cellStart[c + 1];
                    for (int k = cellStart[c]; k < end; k++)
                    {
                        int j = cellItems[k]; if (j == i) continue;
                        Vector2 d = pi - pred[j];
                        float r2 = d.x * d.x + d.y * d.y;
                        if (r2 >= h2 || r2 < 1e-12f) continue;
                        float q = h2 - r2; float w = q * q * q;
                        acc += vCon[j] * w; wSum += w;
                    }
                }
            }
            if (wSum > 1e-12f) vc += (acc / wSum - vc) * visc;
            vOut[i] = vc;
        }

        // ---- 5. hand the result to the engine as ONE batched force.
        // The engine will add gravity itself, so we only ask for the difference to the gravity-only prediction.
        int nf = 0;
        for (int i = 0; i < n; i++)
        {
            Vector2 dv = vOut[i] - vPred[i];
            if (dv.x * dv.x + dv.y * dv.y < 1e-10f) continue;
            Vector2 f = dv * (mass[i] * invDt);
            if (!Finite(f)) continue;
            PhysicsBody.BatchForce bf = new PhysicsBody.BatchForce(bodies[i]);
            bf.ApplyForceToCenter(f);
            forceBatch[nf++] = bf;
        }
        if (nf > 0) PhysicsBody.SetBatchForce(new System.ReadOnlySpan<PhysicsBody.BatchForce>(forceBatch, 0, nf));

        diagAvgDensityRatio = sumRatio / n;
        diagCompressed = compressed;
        diagStepMs = (float)stepWatch.Elapsed.TotalMilliseconds;
        if (showLogs && (++stepCounter % 100) == 0)
            Debug.Log($"[PbfFluidLite] n={n} avg rho/rho0={diagAvgDensityRatio:F2} compressed={compressed} cost={diagStepMs:F2} ms");
    }

    // ================================================================== neighbour grid (counting sort, zero allocation)
    void BuildGrid(Vector2[] p, int n)
    {
        float cs = Mathf.Max(cellSize, smoothingRadius, 0.05f);
        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
        bool any = false;
        for (int i = 0; i < n; i++)
        {
            Vector2 q = p[i];
            if (!Finite(q)) continue;
            if (q.x < minX) minX = q.x;
            if (q.y < minY) minY = q.y;
            if (q.x > maxX) maxX = q.x;
            if (q.y > maxY) maxY = q.y;
            any = true;
        }
        if (!any)
        {
            cellsX = cellsY = 0;
            for (int i = 0; i < n; i++) cellOf[i] = -1;
            return;
        }

        // If the particles are spread very wide, enlarge the cells until the grid fits. Still correct: cells >= h.
        double w = (double)maxX - minX, ht = (double)maxY - minY;
        int cx, cy;
        for (; ; )
        {
            double fx = w / cs + 1.0, fy = ht / cs + 1.0;
            if (fx * fy <= MaxCells) { cx = (int)fx; cy = (int)fy; break; }
            cs *= 1.5f;
        }
        effectiveCell = cs; invCell = 1f / cs; gridMin = new Vector2(minX, minY);
        cellsX = cx; cellsY = cy;

        int total = cx * cy;
        if (cellStart == null || cellStart.Length < total + 1)
        {
            int cap = Mathf.NextPowerOfTwo(total + 1);
            cellStart = new int[cap];
            cellFill = new int[cap];
        }
        else System.Array.Clear(cellStart, 0, total + 1);

        for (int i = 0; i < n; i++)
        {
            Vector2 q = p[i];
            if (!Finite(q)) { cellOf[i] = -1; continue; }
            int gx = (int)((q.x - minX) * invCell);
            int gy = (int)((q.y - minY) * invCell);
            if (gx >= cx) gx = cx - 1;
            if (gy >= cy) gy = cy - 1;
            int c = gy * cx + gx;
            cellOf[i] = c;
            cellStart[c + 1]++;
        }
        for (int c = 0; c < total; c++) cellStart[c + 1] += cellStart[c];
        System.Array.Copy(cellStart, cellFill, total);
        for (int i = 0; i < n; i++)
        {
            int c = cellOf[i];
            if (c >= 0) cellItems[cellFill[c]++] = i;
        }
    }

    // ================================================================== bookkeeping
    void EnsureCapacity(int need)
    {
        if (bodies != null && need <= bodies.Length) return;
        int cap = Mathf.Max(64, need, bodies == null ? 0 : bodies.Length * 2);
        System.Array.Resize(ref bodies, cap);
        System.Array.Resize(ref typeOf, cap);
        System.Array.Resize(ref baseGs, cap);
        System.Array.Resize(ref appliedGs, cap);
        System.Array.Resize(ref mass, cap);
        System.Array.Resize(ref pred, cap);
        System.Array.Resize(ref pos0, cap);
        System.Array.Resize(ref vPred, cap);
        System.Array.Resize(ref vCon, cap);
        System.Array.Resize(ref vOut, cap);
        System.Array.Resize(ref delta, cap);
        System.Array.Resize(ref lambda, cap);
        System.Array.Resize(ref rho0Arr, cap);
        System.Array.Resize(ref stiffArr, cap);
        System.Array.Resize(ref viscArr, cap);
        System.Array.Resize(ref cellOf, cap);
        System.Array.Resize(ref cellItems, cap);
        System.Array.Resize(ref forceBatch, cap);
    }

    void RemoveAt(int i, bool restoreGravity)
    {
        PhysicsBody b = bodies[i];
        if (restoreGravity && b.isValid) b.gravityScale = baseGs[i];
        indexOf.Remove(b);
        int last = count - 1;
        if (i != last)
        {
            bodies[i] = bodies[last]; typeOf[i] = typeOf[last];
            baseGs[i] = baseGs[last]; appliedGs[i] = appliedGs[last]; mass[i] = mass[last];
            indexOf[bodies[i]] = i;
        }
        bodies[last] = default;
        count--;
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
    // Iterates integer lattice indices so the origin is ALWAYS a lattice point (the old float loop only
    // did that when h was an exact multiple of the spacing).
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
        float rho0 = Mathf.Max(0.001f, referenceMass * CurrentLatticeSum(Mathf.Max(0.05f, smoothingRadius)));
        if (types == null || types.Count == 0) types = new List<FluidType>() { new FluidType() };
        foreach (var ft in types) if (ft != null) ft.restDensity = rho0 * ft.restDensityScale;
        return rho0;
    }

    // ================================================================== gizmos
    void DrawGridLines()
    {
        if (cellsX <= 0 || cellsY <= 0) return;
        Vector2 min = gridMin;
        Vector2 max = gridMin + new Vector2(cellsX * effectiveCell, cellsY * effectiveCell);
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.9f);
        Gizmos.DrawLine(new Vector3(min.x, min.y, 0f), new Vector3(max.x, min.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, min.y, 0f), new Vector3(max.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, max.y, 0f), new Vector3(min.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(min.x, max.y, 0f), new Vector3(min.x, min.y, 0f));
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.35f);
        for (int i = 1; i < cellsX && i < 200; i++)
        { float x = min.x + i * effectiveCell; Gizmos.DrawLine(new Vector3(x, min.y, 0f), new Vector3(x, max.y, 0f)); }
        for (int i = 1; i < cellsY && i < 200; i++)
        { float y = min.y + i * effectiveCell; Gizmos.DrawLine(new Vector3(min.x, y, 0f), new Vector3(max.x, y, 0f)); }
    }
    void OnDrawGizmos() { if (!showGridGizmo || !drawGridWhenUnselected) return; DrawGridLines(); }
    void OnDrawGizmosSelected() { if (!showGridGizmo) return; DrawGridLines(); }
}
