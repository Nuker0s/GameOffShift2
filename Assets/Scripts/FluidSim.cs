using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

/// <summary>
/// FluidSim - Burst + Jobs SPH over a uniform grid, PhysicsCore2D-driven particles.
///
/// INTEGRATION MODEL (PhysicsCore2D is the boss):
///   This script NEVER integrates positions/velocities itself. Every step it only
///   computes SPH forces from PhysicsCore state (body.position / body.linearVelocity /
///   body.mass) and calls body.ApplyForceToCenter(). Gravity, damping, collisions and
///   integration all stay inside PhysicsCore2D. There is no fallback velocity estimator,
///   no manual damping compensation, no parallel integrator in here.
/// GRID:
///   Full rebuild every step (counting-sort into cellStart/cellEnd + entries), feeds
///   Burst jobs directly. UpdateParticleCell() is kept only for external queries.
/// MASS vs (shape) DENSITY:
///   SPH uses body.mass (authoritative). Shape density only matters at spawn time when
///   PhysicsCore auto-computes mass from shapes (massConfiguration / ApplyMassFromShapes).
///   Changing a shape density later does NOT change SPH unless the body mass changes.
///   Rest density (restDensity) is an SPH parameter in units of (mass * kernelWeight),
///   NOT a PhysicsCore density.
/// KERNEL CURVE:
///   Physics q = r/h (0=centre contact, 1=edge of support). The inspector curve can be
///   mirrored with mirrorCurveX so that inspector X matches curve presets:
///   when mirrored, inspector left = edge (q=1), inspector right = centre (q=0).
///   Baking samples Evaluate(mirrored ? 1-q : q). Gradients are finite-differenced
///   from the baked LUT so the sign stays correct either way.
/// CLUSTERING NOTES:
///   Negative pressure (rho &lt; rest) used to flip the pressure term into attraction and
///   freeze clumps. Now clamped to >= 0 unless allowNegativePressure is set, and the
///   pressure force uses the symmetric mj*(pA/rhoA^2 + pB/rhoB^2) form.
///   A weak mass-scaled core nudge handles exact overlaps where the smooth gradient -> 0;
///   it is deliberately weaker than SPH pressure. If the sim is driven by the core term,
///   restDensity/gasConstant are mistuned — run Calibrate Rest Density From Curve.
/// FORCE BALANCE (why the curve matters now):
///   With normalizeKernel, W(0)=1 so density ~= mass-weighted neighbour count and the
///   curve SHAPE (not its absolute scale) sets pressure via gradient steepness.
///   Pressure only exists where rho > restDensity, so keep rest near the calibrated hex
///   packing value; far above it mutes everything, far below it makes everything explode.
/// </summary>
[DefaultExecutionOrder(PhysicsExecutionOrder.First)]
public class FluidSim : MonoBehaviour
{
    public enum ParticleTypeSource { None, GravityScale, LinearDamping, GravityPlusDamping }

    [Header("Grid Definition (predefined size + resolution)")]
    [Tooltip("World-space bottom-left corner of the grid.")]
    public Vector2 gridOrigin = new Vector2(-10f, -10f);
    [Tooltip("World-space size of the grid (width, height).")]
    public Vector2 gridSize = new Vector2(20f, 20f);
    [Tooltip("Resolution: world size of one square cell. Keep == smoothingRadius for 3x3 search (fastest).")]
    [Min(0.05f)] public float cellSize = 0.6f;

    [Header("Grid Debug Rendering")]
    [Tooltip("Master switch for grid gizmo rendering (Scene + selected).")]
    public bool showGridGizmo = true;
    [Tooltip("Draw grid even when object is NOT selected (OnDrawGizmos). Off = selected only.")]
    public bool drawGridWhenUnselected = false;

    [Header("Smoothing Kernel (AnimationCurve -> LUT)")]
    [Tooltip("Physics q = r/h (0 = centre contact, 1 = edge of support). With mirrorCurveX ON, inspector X is reversed: left = edge, right = centre (matches curve presets).")]
    public AnimationCurve smoothingCurve = new AnimationCurve(
        new Keyframe(0f, 0f, -0.8f, -0.8f),
        new Keyframe(0.3f, 0.28f, -1.4f, -1.4f),
        new Keyframe(0.65f, 0.72f, -1.2f, -1.2f),
        new Keyframe(1f, 1f, 0f, 0f));
    [Tooltip("Reverse inspector X so presets read left=edge, right=centre. Physics sampling still uses q=r/h internally. NOTE: author the curve in INSPECTOR orientation (left=edge/low, right=centre/high); baking mirrors it to physics q.")]
    public bool mirrorCurveX = true;
    [Tooltip("LUT entries baked from the curve. Higher = smoother gradients, slightly slower bake.")]
    [Range(16, 256)] public int kernelLutSize = 64;
    [Tooltip("Re-bake LUT automatically when curve edits in Inspector.")]
    public bool autoRebakeKernel = true;
    [Header("SPH")]
    [Tooltip("Smoothing radius h. Neighbours within h contribute. Keep ~2.5-3x particle spacing, ~3-4x particle radius.")]
    [Min(0.05f)] public float smoothingRadius = 0.6f;
    [Tooltip("SPH particle mass used ONLY when useBodyMass is OFF. When ON, body.mass (PhysicsCore) is authoritative.")]
    [Min(0.001f)] public float particleMass = 1f;
    [Tooltip("SPH rest density in (bodyMass * kernelWeight) units. NOT a PhysicsCore shape density. Pressure engages only above this value. Default tuned for mass=1, spacing=0.25, h=0.6 with normalized kernel.")]
    [Min(0.01f)] public float restDensity = 2.5f;
    [Tooltip("Pressure stiffness k. Pressure force also scales with kernel gradient, so steeper curve = stronger response at same k.")]
    public float gasConstant = 150f;
    [Tooltip("Viscosity coefficient mu, scaled by neighbour count (XSPH-style): effective damping per step ~ viscosity * neighbours / rho. Start 0.5-2. PhysicsCore linearDamping still applies on top in the engine.")]
    public float viscosity = 0.8f;
    [Tooltip("Maximum neighbours sampled per particle per step (0 = unlimited). Caps cost in dense piles; excess neighbours are skipped beyond this count.")]
    [Min(0)] public int maxNeighboursPerParticle = 32;
    [Tooltip("Normalize baked kernel so self-weight W(0) = 1. Density then reads as (mass-weighted effective neighbour count), making restDensity intuitive and curve-shape changes meaningful instead of rescaling everything.")]
    public bool normalizeKernel = true;
    [Tooltip("Clamp per-particle SPH force magnitude.")]
    [Min(0f)] public float maxForce = 500f;
    public bool enablePressure = true;
    public bool enableViscosity = true;
    [Tooltip("Allow p<0 when rho<rest (cohesive/tensile). OFF clamps p>=0 which strongly reduces permanent clumps.")]
    public bool allowNegativePressure = false;
    [Header("Anti-Clump (short-range overlap safety net — OFF by default)")]
    [Tooltip("Last-resort push for near-exact overlaps where the smooth kernel gradient -> 0. ON with gentle defaults so small clusters visibly separate; SPH pressure still does most work.")]
    public bool enableCoreRepulsion = true;
    [Tooltip("Core radius as fraction of h. Only pairs closer than this get the safety push.")]
    [Range(0.05f, 1f)] public float coreRadiusFraction = 0.3f;
    [Tooltip("Core stiffness scale. Actual push = stiffness * massPair * (1 - r/coreR), deliberately weaker than SPH pressure at normal densities.")]
    public float coreStiffness = 25f;
    [Tooltip("Guarantee a minimum repulsion slope for close pairs: effective |gradW| is at least minGradientFrac * (W0/h) for r < minGradientRange*h. Fixes the flat-centre problem where smooth-curve gradients -> 0 near contact and pressure goes mute. 0 = pure curve gradient.")]
    [Range(0f, 2f)] public float minCentralGradient = 1f;
    [Tooltip("Range (fraction of h) over which the minimum central gradient applies.")]
    [Range(0.05f, 1f)] public float minCentralRange = 0.5f;
    public bool enableSPH = true;
    [Tooltip("Apply SPH in FixedUpdate (runs BEFORE PhysicsCore simulation step thanks to execution order).")]
    public bool runInFixedUpdate = true;

    [Header("PhysicsCore2D particle properties (no custom types)")]
    [Tooltip("Read per-particle mass from body.mass (authoritative). OFF uses particleMass fallback for all.")]
    public bool useBodyMass = true;
    [Tooltip("Type key source for rendering/grouping only. Physics uses real body values.")]
    public ParticleTypeSource particleTypeSource = ParticleTypeSource.GravityPlusDamping;
    [Tooltip("Quantization step for the derived type key.")]
    [Min(0.001f)] public float typeQuantize = 0.05f;

    [Header("Jobs / Burst")]
    [Tooltip("Run density + force passes as Burst IJobParallelFor. Off = managed fallback.")]
    public bool useBurstJobs = true;
    [Tooltip("Batch size for parallel jobs.")]
    [Min(1)] public int jobBatchSize = 64;

    [Header("Optional Auto-Sync")]
    [Tooltip("If assigned, bodies in this BenchMark are auto-registered each physics step.")]
    public BenchMark autoSyncSource;
    public bool autoSyncRemoveMissing = false;

    int _cellsX = 1, _cellsY = 1;
    readonly List<PhysicsBody> _particles = new List<PhysicsBody>(256);
    readonly Dictionary<PhysicsBody, int> _particleCell = new Dictionary<PhysicsBody, int>(256);
    List<PhysicsBody>[] _cells;
    readonly List<float> _density = new List<float>(256);
    readonly List<float> _pressure = new List<float>(256);
    readonly List<Vector2> _sphForce = new List<Vector2>(256);
    readonly Dictionary<PhysicsBody, int> _particleIndex = new Dictionary<PhysicsBody, int>(256);
    float[] _kernelLut = new float[64];
    float[] _kernelGradLut;
    NativeArray<float> _kernelNative;
    NativeArray<float> _kernelGradNative;
    bool _kernelDirty = true;
    NativeArray<float2> _posNative;
    NativeArray<float2> _velNative;
    NativeArray<float> _massNative;
    NativeArray<float> _densNative;
    NativeArray<float> _pressNative;
    NativeArray<float2> _forceNative;
    NativeArray<int> _cellStartNative;
    NativeArray<int> _cellEndNative;
    NativeArray<int> _entriesNative;
    // Burst grid scratch: cell-of-particle + fill cursor (counting sort, no managed Lists).
    NativeArray<int> _cellOfNative;
    NativeArray<int> _cellCursorNative;
    int _nativeCapP = 0, _nativeCapC = 0;
    // Managed _cells lists are only fresh when Burst is off (or after a lazy rebuild
    // for queries). Hot path never touches them.
    bool _managedCellsStale = true;
    int _curveHash = 0;
    public int CellsX => _cellsX;
    public int CellsY => _cellsY;
    public int ParticleCount => _particles.Count;
    public IReadOnlyList<PhysicsBody> Particles => _particles;

    void Awake() { BuildGrid(); BakeKernel(); }
    void OnDestroy() { DisposeNatives(); }

    void OnValidate()
    {
        cellSize = Mathf.Max(0.05f, cellSize);
        gridSize.x = Mathf.Max(cellSize, gridSize.x);
        gridSize.y = Mathf.Max(cellSize, gridSize.y);
        smoothingRadius = Mathf.Max(0.05f, smoothingRadius);
        particleMass = Mathf.Max(0.001f, particleMass);
        restDensity = Mathf.Max(0.01f, restDensity);
        coreRadiusFraction = Mathf.Clamp(coreRadiusFraction, 0.05f, 1f);
        kernelLutSize = Mathf.Clamp(kernelLutSize, 16, 256);
        jobBatchSize = Mathf.Max(1, jobBatchSize);
        maxNeighboursPerParticle = Mathf.Max(0, maxNeighboursPerParticle);
        _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cellSize));
        _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cellSize));
        // AnimationCurve inspector edits don't always trip autoRebake by themselves,
        // so hash the curve state and force a rebake when it changes.
        int h = CurveHash();
        if (h != _curveHash) { _curveHash = h; _kernelDirty = true; }
        else if (autoRebakeKernel) _kernelDirty = true;
        if (Application.isPlaying)
        {
            if (_kernelDirty) BakeKernel();
            BuildGrid();
        }
    }
    int CurveHash()
    {
        unchecked
        {
            int h = 17;
            h = h * 31 + kernelLutSize;
            h = h * 31 + (mirrorCurveX ? 1 : 0);
            h = h * 31 + (normalizeKernel ? 1 : 0);
            if (smoothingCurve != null)
            {
                var keys = smoothingCurve.keys;
                h = h * 31 + smoothingCurve.length;
                for (int i = 0; i < keys.Length; i++)
                {
                    h = h * 31 + keys[i].time.GetHashCode();
                    h = h * 31 + keys[i].value.GetHashCode();
                    h = h * 31 + keys[i].inTangent.GetHashCode();
                    h = h * 31 + keys[i].outTangent.GetHashCode();
                }
            }
            return h;
        }
    }

    // Runs BEFORE PhysicsCore2D simulation (see DefaultExecutionOrder(First)):
    // forces applied here are integrated by the engine this same step.
    void FixedUpdate() { if (runInFixedUpdate) StepPhysics(); }

    public void StepPhysics()
    {
        if (_cells == null) BuildGrid();
        if (autoSyncSource != null) SyncFromBenchmark(autoSyncSource, autoSyncRemoveMissing);
        if (_kernelDirty) BakeKernel();
        if (!enableSPH || _particles.Count == 0) return;
        PruneNulls();
        EnsureSphCapacity();
        RebuildGridFull();
        if (useBurstJobs)
        {
            try { ApplySPHBurst(); return; }
            catch (System.Exception e) { Debug.LogWarning("[FluidSim] Burst path failed, fallback: " + e.Message); }
        }
        ApplySPHManaged();
    }
    public void BakeKernel()
    {
        int n = Mathf.Clamp(kernelLutSize, 16, 256);
        if (_kernelLut == null || _kernelLut.Length != n) _kernelLut = new float[n];
        if (smoothingCurve == null || smoothingCurve.length == 0)
            // Fallback in INSPECTOR orientation (left=edge/low, right=centre/high);
            // the mirror in the bake loop converts to physics q.
            smoothingCurve = new AnimationCurve(
                new Keyframe(0f, 0f, 0f, 2f),
                new Keyframe(1f, 1f, 2f, 0f));
        float peak = 0f;
        for (int i = 0; i < n; i++)
        {
            float q = (float)i / (n - 1);
            // Inspector X is mirrored when mirrorCurveX: inspector 0 = edge (q=1),
            // inspector 1 = centre (q=0). Physics sampling always uses q = r/h,
            // where LUT[0] = centre (W max) and LUT[n-1] = edge (W = 0).
            float sampleX = mirrorCurveX ? 1f - q : q;
            float w = Mathf.Max(0f, smoothingCurve.Evaluate(sampleX));
            _kernelLut[i] = w;
            if (w > peak) peak = w;
        }
        // Normalize so self-weight W(0) = 1: density then reads as mass-weighted
        // effective neighbour count, and reshaping the curve changes the *profile*
        // instead of silently rescaling all densities/pressures.
        if (normalizeKernel && peak > 1e-6f)
            for (int i = 0; i < n; i++) _kernelLut[i] /= peak;
        _kernelLut[n - 1] = 0f;
        _kernelGradLut = new float[n];
        for (int i = 0; i < n; i++)
        {
            int a = Mathf.Max(0, i - 1), b = Mathf.Min(n - 1, i + 1);
            float dq = (b - a) / (float)(n - 1);
            _kernelGradLut[i] = dq > 1e-6f ? (_kernelLut[b] - _kernelLut[a]) / dq : 0f;
        }
        if (_kernelNative.IsCreated) _kernelNative.Dispose();
        if (_kernelGradNative.IsCreated) _kernelGradNative.Dispose();
        _kernelNative = new NativeArray<float>(_kernelLut, Allocator.Persistent);
        _kernelGradNative = new NativeArray<float>(_kernelGradLut, Allocator.Persistent);
        _kernelDirty = false;
    }

    public void MarkKernelDirty() { _kernelDirty = true; }

    static float SampleLut(float[] lut, float q)
    {
        if (lut == null || lut.Length == 0) return 0f;
        q = Mathf.Clamp01(q);
        float f = q * (lut.Length - 1);
        int i = Mathf.Min((int)f, lut.Length - 2);
        return Mathf.Lerp(lut[i], lut[i + 1], f - i);
    }
    public float SampleKernel(float q) { return SampleLut(_kernelLut, q); }
    public float SampleKernelGradient(float q) { return SampleLut(_kernelGradLut, q); }

    /// <summary>SPH particle mass. body.mass (PhysicsCore) is authoritative when useBodyMass is on.</summary>
    public float GetBodyMass(PhysicsBody b)
    {
        if (useBodyMass && IsValidBody(b))
        {
            float m = b.mass;
            if (m > 1e-6f && m < 1e6f) return m;
        }
        return particleMass;
    }
    static bool IsValidBody(PhysicsBody b)
    {
        // PhysicsBody is a struct handle: check validity directly (no exceptions in hot path).
        return b.isValid;
    }
    public Vector2 GetBodyVelocity(PhysicsBody b)
    {
        // PhysicsCore owns velocity. No fallback estimator: if the body is gone, return zero.
        if (!b.isValid) return Vector2.zero;
        Vector2 v = b.linearVelocity;
        return v.sqrMagnitude < 1e7f ? v : Vector2.zero;
    }
    public float GetBodyDamping(PhysicsBody b)
    {
        if (!IsValidBody(b)) return 0f;
        return Mathf.Max(0f, b.linearDamping);
    }
    public float GetBodyGravity(PhysicsBody b)
    {
        if (!IsValidBody(b)) return 1f;
        return b.gravityScale;
    }
    public float GetParticleTypeKey(PhysicsBody b)
    {
        if (!IsValidBody(b)) return 0f;
        float q = Mathf.Max(typeQuantize, 0.001f);
        float g = GetBodyGravity(b), d = GetBodyDamping(b);
        switch (particleTypeSource)
        {
            case ParticleTypeSource.GravityScale: return Mathf.Round(g / q) * q;
            case ParticleTypeSource.LinearDamping: return Mathf.Round(d / q) * q;
            case ParticleTypeSource.GravityPlusDamping:
                return Mathf.Round(g / q) * q * 10f + Mathf.Round(d / q) * q;
            default: return 0f;
        }
    }
    #region Grid (full rebuild every step)
    public void BuildGrid()
    {
        _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / Mathf.Max(0.05f, cellSize)));
        _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / Mathf.Max(0.05f, cellSize)));
        _cells = new List<PhysicsBody>[_cellsX * _cellsY];
        for (int i = 0; i < _cells.Length; i++) _cells[i] = new List<PhysicsBody>(8);
        RebuildManagedCells();
    }
    public void ClearCells()
    {
        if (_cells == null) return;
        for (int i = 0; i < _cells.Length; i++) _cells[i].Clear();
    }
    public bool WorldToCell(Vector2 wp, out int cx, out int cy)
    {
        cx = Mathf.FloorToInt((wp.x - gridOrigin.x) / cellSize);
        cy = Mathf.FloorToInt((wp.y - gridOrigin.y) / cellSize);
        return cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY;
    }
    public int CellToIndex(int cx, int cy) { return cy * _cellsX + cx; }
    public bool IsValidCell(int cx, int cy) { return cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY; }
    public int GetCellIndexForPosition(Vector2 wp)
    {
        int cx, cy;
        if (!WorldToCell(wp, out cx, out cy)) return -1;
        return CellToIndex(cx, cy);
    }
    public int GetParticleCellIndex(PhysicsBody b)
    {
        // Live lookup: no per-step dictionary churn in the hot loop.
        for (int i = 0; i < _particles.Count; i++)
            if (_particles[i].Equals(b))
                return GetCellIndexForPosition(_particles[i].position);
        return -1;
    }
    public int NeighbourCellSpan()
    {
        if (cellSize <= 0f) return 1;
        return Mathf.Max(1, Mathf.CeilToInt(smoothingRadius / cellSize));
    }
    void PruneNulls()
    {
        // Swap-remove: O(n), no shifting, no exceptions, no allocs.
        bool removed = false;
        for (int i = _particles.Count - 1; i >= 0; i--)
        {
            if (!_particles[i].isValid)
            {
                int last = _particles.Count - 1;
                PhysicsBody dead = _particles[i];
                _particleCell.Remove(dead);
                if (i != last) _particles[i] = _particles[last];
                _particles.RemoveAt(last);
                removed = true;
            }
        }
        if (removed) _managedCellsStale = true;
    }
    void EnsureSphCapacity()
    {
        int n = _particles.Count;
        if (_density.Count < n)
        {
            int add = n - _density.Count;
            for (int k = 0; k < add; k++) { _density.Add(0f); _pressure.Add(0f); _sphForce.Add(Vector2.zero); }
        }
        else if (_density.Count > n)
        {
            _density.RemoveRange(n, _density.Count - n);
            _pressure.RemoveRange(n, _pressure.Count - n);
            _sphForce.RemoveRange(n, _sphForce.Count - n);
        }
        // Managed fallback needs body->index; Burst path does NOT (entries store indices).
        // Rebuild only when Burst is off. When Burst is on, clear so stale indices can
        // never return a wrong row (queries fall back to linear scan, hot path unaffected).
        _particleIndex.Clear();
        if (!useBurstJobs)
        {
            for (int i = 0; i < n; i++) _particleIndex[_particles[i]] = i;
        }
    }
    void EnsureNative(int pCount, int cCount)
    {
        // Grow-only with doubling: steady-state performs ZERO allocs/disposes per step.
        if (pCount > _nativeCapP)
        {
            int cap = Mathf.Max(Mathf.Max(64, pCount), _nativeCapP * 2);
            if (_posNative.IsCreated) _posNative.Dispose();
            if (_velNative.IsCreated) _velNative.Dispose();
            if (_massNative.IsCreated) _massNative.Dispose();
            if (_densNative.IsCreated) _densNative.Dispose();
            if (_pressNative.IsCreated) _pressNative.Dispose();
            if (_forceNative.IsCreated) _forceNative.Dispose();
            if (_entriesNative.IsCreated) _entriesNative.Dispose();
            if (_cellOfNative.IsCreated) _cellOfNative.Dispose();
            _posNative = new NativeArray<float2>(cap, Allocator.Persistent);
            _velNative = new NativeArray<float2>(cap, Allocator.Persistent);
            _massNative = new NativeArray<float>(cap, Allocator.Persistent);
            _densNative = new NativeArray<float>(cap, Allocator.Persistent);
            _pressNative = new NativeArray<float>(cap, Allocator.Persistent);
            _forceNative = new NativeArray<float2>(cap, Allocator.Persistent);
            _entriesNative = new NativeArray<int>(cap, Allocator.Persistent);
            _cellOfNative = new NativeArray<int>(cap, Allocator.Persistent);
            _nativeCapP = cap;
        }
        if (cCount > _nativeCapC)
        {
            int cap = Mathf.Max(Mathf.Max(64, cCount), _nativeCapC * 2);
            if (_cellStartNative.IsCreated) _cellStartNative.Dispose();
            if (_cellEndNative.IsCreated) _cellEndNative.Dispose();
            if (_cellCursorNative.IsCreated) _cellCursorNative.Dispose();
            _cellStartNative = new NativeArray<int>(cap, Allocator.Persistent);
            _cellEndNative = new NativeArray<int>(cap, Allocator.Persistent);
            _cellCursorNative = new NativeArray<int>(cap, Allocator.Persistent);
            _nativeCapC = cap;
        }
    }
    void DisposeNatives()
    {
        if (_posNative.IsCreated) _posNative.Dispose();
        if (_velNative.IsCreated) _velNative.Dispose();
        if (_massNative.IsCreated) _massNative.Dispose();
        if (_densNative.IsCreated) _densNative.Dispose();
        if (_pressNative.IsCreated) _pressNative.Dispose();
        if (_forceNative.IsCreated) _forceNative.Dispose();
        if (_entriesNative.IsCreated) _entriesNative.Dispose();
        if (_cellOfNative.IsCreated) _cellOfNative.Dispose();
        if (_cellStartNative.IsCreated) _cellStartNative.Dispose();
        if (_cellEndNative.IsCreated) _cellEndNative.Dispose();
        if (_cellCursorNative.IsCreated) _cellCursorNative.Dispose();
        if (_kernelNative.IsCreated) _kernelNative.Dispose();
        if (_kernelGradNative.IsCreated) _kernelGradNative.Dispose();
        _nativeCapP = _nativeCapC = 0;
    }
    // Fast native grid: O(n + cells), no managed per-cell Lists, no Dictionary lookups.
    // Fills pos/vel/mass + counting-sort entries in one pass. Managed _cells lists are
    // only rebuilt for the managed fallback / external queries (lazy).
    void RebuildGridFull()
    {
        int n = _particles.Count;
        int cells = _cellsX * _cellsY;
        EnsureNative(n, cells);
        float ox = gridOrigin.x, oy = gridOrigin.y;
        float cs = Mathf.Max(0.05f, cellSize);
        float invCs = 1f / cs;
        // Zero counts (cells is typically a few hundred — cheap memset-style loop).
        for (int c = 0; c < cells; c++) _cellStartNative[c] = 0;
        // Pass 1: gather state + count per cell.
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            Vector2 bp = b.position;
            _posNative[i] = new float2(bp.x, bp.y);
            Vector2 bv = b.linearVelocity;
            _velNative[i] = new float2(bv.x, bv.y);
            float m = b.mass;
            _massNative[i] = (m > 1e-6f && m < 1e6f) ? m : particleMass;
            int cx = (int)((bp.x - ox) * invCs);
            int cy = (int)((bp.y - oy) * invCs);
            int cell = (cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY) ? (cy * _cellsX + cx) : -1;
            _cellOfNative[i] = cell;
            if (useBodyMass == false) _massNative[i] = particleMass;
            if (cell >= 0) _cellStartNative[cell]++;
            // Per-step cell value goes stale as bodies move; membership stays valid.
            // RebuildGridFull deliberately does NOT touch this dictionary (no hashing
            // in the hot loop). Cell index is derived live via GetParticleCellIndex.
        }
        // Prefix sum: start[c] = running total; end[c] computed after.
        int cursor = 0;
        for (int c = 0; c < cells; c++)
        {
            int cnt = _cellStartNative[c];
            _cellStartNative[c] = cursor;
            _cellCursorNative[c] = cursor;
            cursor += cnt;
            _cellEndNative[c] = cursor;
        }
        // Pass 2: scatter particle indices into entries.
        for (int i = 0; i < n; i++)
        {
            int cell = _cellOfNative[i];
            if (cell >= 0) _entriesNative[_cellCursorNative[cell]++] = i;
        }
        // Managed cell lists: only rebuilt lazily for external queries.
        // Hot path never touches them, so mark stale instead of rebuilding.
        _managedCellsStale = true;
    }
    // Lazy rebuild of managed per-cell Lists (fallback + GetNeighbours/GetCellContents).
    void RebuildManagedCells()
    {
        if (_cells == null || _cells.Length != _cellsX * _cellsY)
        {
            _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / Mathf.Max(0.05f, cellSize)));
            _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / Mathf.Max(0.05f, cellSize)));
            _cells = new List<PhysicsBody>[_cellsX * _cellsY];
            for (int i = 0; i < _cells.Length; i++) _cells[i] = new List<PhysicsBody>(8);
        }
        ClearCells();
        _particleCell.Clear();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            int idx = GetCellIndexForPosition(b.position);
            _particleCell[b] = idx;
            if (idx >= 0) _cells[idx].Add(b);
        }
        _managedCellsStale = false;
    }
    void EnsureManagedCellsFresh()
    {
        if (_managedCellsStale) RebuildManagedCells();
    }
    #endregion
    [BurstCompile]
    struct DensityPressureJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> pos;
        [ReadOnly] public NativeArray<float> mass;
        [ReadOnly] public NativeArray<int> cellStart;
        [ReadOnly] public NativeArray<int> cellEnd;
        [ReadOnly] public NativeArray<int> entries;
        [ReadOnly] public NativeArray<float> kernel;
        public int cellsX, cellsY, span, lutN;
        public float2 origin;
        public float invCell, h, h2, selfW, gasK, restRho;
        public int allowNeg;
        public NativeArray<float> density;
        public NativeArray<float> pressure;
        public void Execute(int i)
        {
            float2 p = pos[i];
            float mi = mass[i];
            float rho = selfW * mi;
            int cx = (int)math.floor((p.x - origin.x) * invCell);
            int cy = (int)math.floor((p.y - origin.y) * invCell);
            if (cx >= 0 && cy >= 0 && cx < cellsX && cy < cellsY)
            {
                int x0 = cx - span; if (x0 < 0) x0 = 0;
                int x1 = cx + span; if (x1 >= cellsX) x1 = cellsX - 1;
                int y0 = cy - span; if (y0 < 0) y0 = 0;
                int y1 = cy + span; if (y1 >= cellsY) y1 = cellsY - 1;
                float invH = 1f / h;
                float lutScale = (float)(lutN - 1);
                for (int ny = y0; ny <= y1; ny++)
                {
                    int rowBase = ny * cellsX;
                    for (int nx = x0; nx <= x1; nx++)
                    {
                        int c = rowBase + nx;
                        for (int e = cellStart[c]; e < cellEnd[c]; e++)
                        {
                            int j = entries[e];
                            if (j == i) continue;
                            float2 d = p - pos[j];
                            float r2 = d.x * d.x + d.y * d.y;
                            if (r2 >= h2) continue;
                            float q = math.sqrt(r2) * invH;
                            float ff = q * lutScale;
                            int k = (int)ff;
                            if (k < 0) k = 0; else if (k >= lutN - 1) k = lutN - 2;
                            float tt = ff - (float)k;
                            float w = kernel[k] + (kernel[k + 1] - kernel[k]) * tt;
                            rho += mass[j] * w;
                        }
                    }
                }
            }
            if (rho < 1e-5f) rho = 1e-5f;
            density[i] = rho;
            float pr = gasK * (rho - restRho);
            if (allowNeg == 0 && pr < 0f) pr = 0f;
            pressure[i] = pr;
        }
    }
    [BurstCompile]
    struct ForceJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> pos;
        [ReadOnly] public NativeArray<float2> vel;
        [ReadOnly] public NativeArray<float> mass;
        [ReadOnly] public NativeArray<float> density;
        [ReadOnly] public NativeArray<float> pressure;
        [ReadOnly] public NativeArray<int> cellStart;
        [ReadOnly] public NativeArray<int> cellEnd;
        [ReadOnly] public NativeArray<int> entries;
        [ReadOnly] public NativeArray<float> kernelGrad;
        public int cellsX, cellsY, span, lutN;
        public float2 origin;
        public float invCell, h, h2, visc, maxF, coreR, coreK;
        public float minGrad, minGradRange, w0;
        public int doP, doV, doCore, maxNeigh;
        public NativeArray<float2> force;
        public void Execute(int i)
        {
            float2 p = pos[i];
            float rhoA = density[i];
            if (rhoA < 1e-5f) rhoA = 1e-5f;
            float pA = pressure[i];
            float2 vi = vel[i];
            float mi = mass[i];
            float2 f = float2.zero;
            int cx = (int)math.floor((p.x - origin.x) * invCell);
            int cy = (int)math.floor((p.y - origin.y) * invCell);
            // Fast path: below rest density with no viscosity/core there is nothing to do.
            if (pA == 0f && doV == 0 && doCore == 0) { force[i] = float2.zero; return; }
            if (cx >= 0 && cy >= 0 && cx < cellsX && cy < cellsY)
            {
                int x0 = cx - span; if (x0 < 0) x0 = 0;
                int x1 = cx + span; if (x1 >= cellsX) x1 = cellsX - 1;
                int y0 = cy - span; if (y0 < 0) y0 = 0;
                int y1 = cy + span; if (y1 >= cellsY) y1 = cellsY - 1;
                float invH = 1f / h;
                float lutScale = (float)(lutN - 1);
                float invRhoA2 = 0f;
                if (doP == 1 && pA != 0f) invRhoA2 = pA / (rhoA * rhoA);
                float needGrad = 0f;
                if (minGrad > 0f && minGradRange > 1e-6f)
                    needGrad = -minGrad * (w0 * invH);
                int countedN = 0;
                bool useCap = maxNeigh > 0;
                for (int ny = y0; ny <= y1; ny++)
                {
                    int rowBase = ny * cellsX;
                    for (int nx = x0; nx <= x1; nx++)
                    {
                        int c = rowBase + nx;
                        for (int e = cellStart[c]; e < cellEnd[c]; e++)
                        {
                            int j = entries[e];
                            if (j == i) continue;
                            float2 delta = p - pos[j];
                            float r2 = delta.x * delta.x + delta.y * delta.y;
                            if (r2 >= h2) continue;
                            if (useCap)
                            {
                                if (countedN >= maxNeigh) goto CapDone;
                                countedN++;
                            }
                            float mj = mass[j];
                            if (r2 < 1e-12f)
                            {
                                // Exactly overlapped: weak deterministic nudge along +X.
                                if (doCore == 1 && coreK > 0f && coreR > 1e-6f)
                                    f.x += coreK * (mi + mj) * 0.5f;
                                continue;
                            }
                            float r = math.sqrt(r2);
                            float invR = 1f / r;
                            float dirx = delta.x * invR;
                            float diry = delta.y * invR;
                            float rhoB = density[j];
                            if (rhoB < 1e-5f) rhoB = 1e-5f;
                            float qq = r * invH;
                            if (qq > 1f) qq = 1f;
                            float ff = qq * lutScale;
                            int k = (int)ff;
                            if (k < 0) k = 0; else if (k >= lutN - 1) k = lutN - 2;
                            float tt = ff - (float)k;
                            float gw = (kernelGrad[k] + (kernelGrad[k + 1] - kernelGrad[k]) * tt) * invH;
                            // Minimum central slope: keeps pressure alive for close pairs.
                            if (needGrad < 0f && r < minGradRange && gw > needGrad) gw = needGrad;
                            // Symmetric form: mj * (pA/rhoA^2 + pB/rhoB^2), mass-aware.
                            if (doP == 1 && (pA != 0f || pressure[j] != 0f))
                            {
                                float s = invRhoA2 + pressure[j] / (rhoB * rhoB);
                                float sc = -mj * s * gw;
                                f.x += sc * dirx;
                                f.y += sc * diry;
                            }
                            if (doV == 1)
                            {
                                float vc = mj * visc / rhoB * math.abs(gw);
                                float2 dv = vel[j] - vi;
                                f.x += vc * dv.x;
                                f.y += vc * dv.y;
                            }
                            // Safety net only: weak mass-scaled push deep inside coreR.
                            if (doCore == 1 && coreK > 0f && coreR > 1e-6f && r < coreR)
                            {
                                float cc = coreK * (mi + mj) * 0.5f * (1f - r / coreR);
                                f.x += dirx * cc;
                                f.y += diry * cc;
                            }
                        }
                    }
                }
            CapDone:;
            }
            if (maxF > 0f)
            {
                float l2 = f.x * f.x + f.y * f.y;
                float mf2 = maxF * maxF;
                if (l2 > mf2) { float s = maxF / math.sqrt(l2); f.x *= s; f.y *= s; }
            }
            force[i] = f;
        }
    }
    float ComputePressure(float rho)
    {
        float p = gasConstant * (rho - restDensity);
        return allowNegativePressure ? p : Mathf.Max(0f, p);
    }
    void ApplySPHBurst()
    {
        int n = _particles.Count;
        int span = NeighbourCellSpan();
        float h = Mathf.Max(0.05f, smoothingRadius);
        float cs = Mathf.Max(0.05f, cellSize);
        float selfW = _kernelLut != null && _kernelLut.Length > 0 ? _kernelLut[0] : 1f;
        float2 org = new float2(gridOrigin.x, gridOrigin.y);
        float invCell = 1f / cs;
        // Fused density+pressure in one Burst pass (no managed loop between passes).
        var dp = new DensityPressureJob
        {
            pos = _posNative, mass = _massNative,
            cellStart = _cellStartNative, cellEnd = _cellEndNative, entries = _entriesNative,
            kernel = _kernelNative, cellsX = _cellsX, cellsY = _cellsY, span = span,
            lutN = _kernelNative.Length,
            origin = org, invCell = invCell,
            h = h, h2 = h * h, selfW = selfW,
            gasK = gasConstant, restRho = restDensity,
            allowNeg = allowNegativePressure ? 1 : 0,
            density = _densNative, pressure = _pressNative
        };
        float coreR = h * Mathf.Clamp01(coreRadiusFraction);
        var fj = new ForceJob
        {
            pos = _posNative, vel = _velNative, mass = _massNative,
            density = _densNative, pressure = _pressNative,
            cellStart = _cellStartNative, cellEnd = _cellEndNative, entries = _entriesNative,
            kernelGrad = _kernelGradNative, cellsX = _cellsX, cellsY = _cellsY, span = span,
            lutN = _kernelGradNative.Length,
            origin = org, invCell = invCell,
            h = h, h2 = h * h, visc = viscosity, maxF = maxForce,
            coreR = coreR, coreK = coreStiffness,
            minGrad = minCentralGradient, minGradRange = minCentralRange * h,
            w0 = selfW,
            doP = enablePressure ? 1 : 0, doV = enableViscosity ? 1 : 0,
            doCore = enableCoreRepulsion ? 1 : 0, maxNeigh = maxNeighboursPerParticle,
            force = _forceNative
        };
        int batch = Mathf.Max(1, jobBatchSize);
        // Chain: force waits on density (single sync point at the end).
        var dh = dp.Schedule(n, batch);
        var fh = fj.Schedule(n, batch, dh);
        fh.Complete();
        // Single copy-out for external queries + force application.
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            _density[i] = _densNative[i];
            _pressure[i] = _pressNative[i];
            float2 f = _forceNative[i];
            Vector2 fv = new Vector2(f.x, f.y);
            _sphForce[i] = fv;
            // PhysicsCore integrates: we only add force at the centre (no torque).
            if (fv.x != 0f || fv.y != 0f) b.ApplyForceToCenter(fv, true);
        }
    }
    void ApplySPHManaged()
    {
        ComputeSPHDensitiesManaged();
        ComputeSPHPressuresManaged();
        ComputeSPHForcesManaged();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            Vector2 f = _sphForce[i];
            if (f.sqrMagnitude >= 1e-10f) b.ApplyForceToCenter(f, true);
        }
    }
    public void ComputeSPHDensitiesManaged()
    {
        // Managed fallback now walks the same native counting-sort grid as Burst
        // (no per-cell List, no Dictionary, no sqrt-then-magnitude, no per-neighbour calls).
        int n = _particles.Count;
        if (n == 0) return;
        EnsureSphCapacity();
        RebuildGridFull();
        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h;
        float invH = 1f / h;
        float selfW = _kernelLut != null && _kernelLut.Length > 0 ? _kernelLut[0] : 1f;
        int span = NeighbourCellSpan();
        int lutN = _kernelLut != null ? _kernelLut.Length : 1;
        float cs = Mathf.Max(0.05f, cellSize);
        float invCell = 1f / cs;
        float ox = gridOrigin.x, oy = gridOrigin.y;
        for (int i = 0; i < n; i++)
        {
            float2 pi = _posNative[i];
            float mi = _massNative[i];
            float rho = mi * selfW;
            int cx = (int)((pi.x - ox) * invCell);
            int cy = (int)((pi.y - oy) * invCell);
            if (cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY)
            {
                int x0 = cx - span; if (x0 < 0) x0 = 0;
                int x1 = cx + span; if (x1 >= _cellsX) x1 = _cellsX - 1;
                int y0 = cy - span; if (y0 < 0) y0 = 0;
                int y1 = cy + span; if (y1 >= _cellsY) y1 = _cellsY - 1;
                for (int ny = y0; ny <= y1; ny++)
                {
                    int rowBase = ny * _cellsX;
                    for (int nx = x0; nx <= x1; nx++)
                    {
                        int c = rowBase + nx;
                        for (int e = _cellStartNative[c]; e < _cellEndNative[c]; e++)
                        {
                            int j = _entriesNative[e];
                            if (j == i) continue;
                            float2 d = pi - _posNative[j];
                            float r2 = d.x * d.x + d.y * d.y;
                            if (r2 >= h2) continue;
                            float q = Mathf.Sqrt(r2) * invH;
                            if (q > 1f) q = 1f;
                            rho += _massNative[j] * SampleKernel(q);
                        }
                    }
                }
            }
            _density[i] = Mathf.Max(rho, 1e-5f);
        }
    }
    public void ComputeSPHPressuresManaged()
    {
        // Clamped by default: rho < rest would otherwise create suction that locks clumps.
        for (int i = 0; i < _particles.Count; i++)
            _pressure[i] = ComputePressure(_density[i]);
    }
    public void ComputeSPHForcesManaged()
    {
        // Managed fallback over the native grid: index-based, hoisted invariants,
        // single sqrt per pair, no Dictionary/property lookups in the inner loop.
        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h;
        float invH = 1f / h;
        float coreR = h * Mathf.Clamp01(coreRadiusFraction);
        float w0m = (_kernelLut != null && _kernelLut.Length > 0) ? _kernelLut[0] : 1f;
        float minR = minCentralRange * h;
        float needBase = -minCentralGradient * (w0m * invH);
        int span = NeighbourCellSpan();
        int n = _particles.Count;
        float cs = Mathf.Max(0.05f, cellSize);
        float invCell = 1f / cs;
        float ox = gridOrigin.x, oy = gridOrigin.y;
        bool doP = enablePressure, doV = enableViscosity, doC = enableCoreRepulsion;
        float visc = viscosity, coreK = coreStiffness, mf = maxForce;
        int cap = maxNeighboursPerParticle;
        for (int i = 0; i < n; i++)
        {
            float2 pi = _posNative[i];
            float2 vi = _velNative[i];
            float mi = _massNative[i];
            float rhoA = Mathf.Max(_density[i], 1e-5f);
            float pA = _pressure[i];
            float invRhoA2 = (doP && pA != 0f) ? pA / (rhoA * rhoA) : 0f;
            float fx = 0f, fy = 0f;
            if (pA != 0f || doV || doC)
            {
                int cx = (int)((pi.x - ox) * invCell);
                int cy = (int)((pi.y - oy) * invCell);
                if (cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY)
                {
                    int x0 = cx - span; if (x0 < 0) x0 = 0;
                    int x1 = cx + span; if (x1 >= _cellsX) x1 = _cellsX - 1;
                    int y0 = cy - span; if (y0 < 0) y0 = 0;
                    int y1 = cy + span; if (y1 >= _cellsY) y1 = _cellsY - 1;
                    int countedN = 0;
                    bool capHit = false;
                    for (int ny = y0; ny <= y1 && !capHit; ny++)
                    {
                        int rowBase = ny * _cellsX;
                        for (int nx = x0; nx <= x1 && !capHit; nx++)
                        {
                            int c = rowBase + nx;
                            for (int e = _cellStartNative[c]; e < _cellEndNative[c]; e++)
                            {
                                int j = _entriesNative[e];
                                if (j == i) continue;
                                float2 delta = pi - _posNative[j];
                                float r2 = delta.x * delta.x + delta.y * delta.y;
                                if (r2 >= h2) continue;
                                if (cap > 0)
                                {
                                    if (countedN >= cap) { capHit = true; break; }
                                    countedN++;
                                }
                                float mj = _massNative[j];
                                if (r2 < 1e-12f)
                                {
                                    if (doC) fx += coreK * (mi + mj) * 0.5f;
                                    continue;
                                }
                                float r = Mathf.Sqrt(r2);
                                float invR = 1f / r;
                                float dirx = delta.x * invR, diry = delta.y * invR;
                                float rhoB = _density[j];
                                if (rhoB < 1e-5f) rhoB = 1e-5f;
                                float pB = _pressure[j];
                                float gw = SampleKernelGradient(r * invH) * invH;
                                if (needBase < 0f && r < minR && r > 1e-6f && gw > needBase) gw = needBase;
                                if (doP && (pA != 0f || pB != 0f))
                                {
                                    float sc = -mj * (invRhoA2 + pB / (rhoB * rhoB)) * gw;
                                    fx += sc * dirx; fy += sc * diry;
                                }
                                if (doV)
                                {
                                    float vc = mj * visc / rhoB * Mathf.Abs(gw);
                                    float2 dv = _velNative[j] - vi;
                                    fx += vc * dv.x; fy += vc * dv.y;
                                }
                                if (doC && coreR > 1e-6f && r < coreR)
                                {
                                    float cc = coreK * (mi + mj) * 0.5f * (1f - r / coreR);
                                    fx += dirx * cc; fy += diry * cc;
                                }
                            }
                        }
                    }
                }
            }
            if (mf > 0f)
            {
                float l2 = fx * fx + fy * fy;
                if (l2 > mf * mf) { float s = mf / Mathf.Sqrt(l2); fx *= s; fy *= s; }
            }
            _sphForce[i] = new Vector2(fx, fy);
        }
    }
    #region Public API (kept compatible)
    public bool Contains(PhysicsBody body)
    {
        if (!IsValidBody(body)) return false;
        // Membership set maintained on Add/Remove/Prune only — O(1), no per-step churn.
        return _particleCell.ContainsKey(body);
    }
    public bool AddParticle(PhysicsBody body)
    {
        if (!IsValidBody(body)) return false;
        if (_particleCell.ContainsKey(body)) return false;
        if (_cells == null) BuildGrid();
        _particles.Add(body);
        _particleCell[body] = GetCellIndexForPosition(body.position);
        _managedCellsStale = true;
        return true;
    }
    public int AddParticles(IEnumerable<PhysicsBody> bodies)
    {
        int added = 0;
        if (bodies == null) return added;
        foreach (var b in bodies) if (AddParticle(b)) added++;
        return added;
    }
    public bool RemoveParticle(PhysicsBody body)
    {
        if (!IsValidBody(body)) return false;
        for (int i = 0; i < _particles.Count; i++)
        {
            if (_particles[i].Equals(body))
            {
                int last = _particles.Count - 1;
                if (i != last) _particles[i] = _particles[last];
                _particles.RemoveAt(last);
                _managedCellsStale = true;
                return true;
            }
        }
        return false;
    }
    public void ClearParticles()
    {
        _particles.Clear(); _particleCell.Clear(); _particleIndex.Clear();
        _density.Clear(); _pressure.Clear(); _sphForce.Clear();
        ClearCells();
        _managedCellsStale = false;
    }
    public void SyncFromBenchmark(BenchMark source, bool removeMissing = false)
    {
        if (source == null || source.Bodies == null) return;
        var src = source.Bodies;
        // O(n) membership via _particleCell (no per-body linear scans, no temp set alloc
        // beyond what the dictionary already gives us).
        bool grew = false;
        for (int i = 0; i < src.Count; i++)
        {
            var b = src[i];
            if (!IsValidBody(b) || _particleCell.ContainsKey(b)) continue;
            if (_cells == null) BuildGrid();
            _particles.Add(b);
            _particleCell[b] = GetCellIndexForPosition(b.position);
            grew = true;
        }
        if (grew) _managedCellsStale = true;
        if (removeMissing)
        {
            var set = new HashSet<PhysicsBody>(src);
            for (int i = _particles.Count - 1; i >= 0; i--)
                if (!set.Contains(_particles[i])) RemoveParticle(_particles[i]);
        }
    }
    public bool UpdateParticleCell(PhysicsBody body)
    {
        // Grid is rebuilt from scratch every step, so there is nothing incremental to do.
        // Kept for API compatibility: report tracked state.
        if (!IsValidBody(body)) return false;
        return _particleCell.ContainsKey(body);
    }
    public void UpdateAllParticleCells() { PruneNulls(); EnsureSphCapacity(); RebuildGridFull(); }
    /// <summary>
    /// Estimate rest density for a hexagonal packing at targetSpacing using the CURRENT
    /// baked kernel, then assign it. Run once after editing smoothingCurve/smoothingRadius:
    /// right-click the component header > Calibrate Rest Density From Curve.
    /// With a normalized kernel, density ~= sum of neighbour weights, so this samples
    /// the 6 ring neighbours at targetSpacing plus self-weight.
    /// </summary>
    [ContextMenu("Calibrate Rest Density From Curve")]
    public void CalibrateRestDensity()
    {
        if (_kernelDirty) BakeKernel();
        float m = particleMass;
        if (useBodyMass && _particles.Count > 0) m = GetBodyMass(_particles[0]);
        m = Mathf.Max(m, 1e-6f);
        // Fall back to spawn spacing if we can read it.
        float spacing = smoothingRadius * 0.75f;
        if (autoSyncSource != null) spacing = Mathf.Max(0.01f, autoSyncSource.spacing);
        float q = Mathf.Clamp01(spacing / Mathf.Max(0.05f, smoothingRadius));
        // 6 hex neighbours at spacing + self.
        float calibrated = m * (SampleKernel(0f) + 6f * SampleKernel(q));
        restDensity = Mathf.Max(0.01f, calibrated * 0.9f);
        Debug.Log($"[FluidSim] Calibrated restDensity={restDensity:F3} (mass={m:F3}, spacing={spacing:F3}, q={q:F3}, W0={SampleKernel(0f):F3}, Wq={SampleKernel(q):F3}). Pressure now engages above this density.");
#if UNITY_EDITOR
        UnityEditor.EditorUtility.SetDirty(this);
#endif
    }
    public void GetNeighbours(PhysicsBody body, List<PhysicsBody> outN, float radius)
    {
        outN.Clear();
        if (!IsValidBody(body)) return;
        int self = -1;
        for (int k = 0; k < _particles.Count; k++)
            if (_particles[k].Equals(body)) { self = k; break; }
        if (self < 0 || !_entriesNative.IsCreated) return;
        float cs = Mathf.Max(0.0001f, cellSize);
        int span = radius <= 0f ? NeighbourCellSpan() : Mathf.Max(1, Mathf.CeilToInt(radius / cs));
        float h = radius > 0f ? radius : Mathf.Max(0.05f, smoothingRadius);
        float r2 = h * h;
        float2 p = _posNative[self];
        float ox = gridOrigin.x, oy = gridOrigin.y;
        float invCell = 1f / Mathf.Max(0.05f, cellSize);
        int cx = (int)((p.x - ox) * invCell);
        int cy = (int)((p.y - oy) * invCell);
        if (cx < 0 || cy < 0 || cx >= _cellsX || cy >= _cellsY) return;
        int x0 = cx - span; if (x0 < 0) x0 = 0;
        int x1 = cx + span; if (x1 >= _cellsX) x1 = _cellsX - 1;
        int y0 = cy - span; if (y0 < 0) y0 = 0;
        int y1 = cy + span; if (y1 >= _cellsY) y1 = _cellsY - 1;
        for (int ny = y0; ny <= y1; ny++)
        {
            int rowBase = ny * _cellsX;
            for (int nx = x0; nx <= x1; nx++)
            {
                int c = rowBase + nx;
                for (int e = _cellStartNative[c]; e < _cellEndNative[c]; e++)
                {
                    int j = _entriesNative[e];
                    if (j == self) continue;
                    float2 d = p - _posNative[j];
                    if (radius > 0f && (d.x * d.x + d.y * d.y) >= r2) continue;
                    outN.Add(_particles[j]);
                }
            }
        }
    }
    public void GetNeighbours3x3(PhysicsBody body, List<PhysicsBody> outN) { GetNeighbours(body, outN, cellSize * 1.5f); }
    public List<PhysicsBody> GetCellContents(int cx, int cy)
    {
        EnsureManagedCellsFresh();
        if (_cells == null || !IsValidCell(cx, cy)) return null;
        return _cells[CellToIndex(cx, cy)];
    }
    public float GetDensity(PhysicsBody b)
    {
        int i;
        if (_particleIndex.TryGetValue(b, out i) && i >= 0 && i < _density.Count) return _density[i];
        // Burst path skips the index rebuild: linear scan fallback (queries only, not hot path).
        for (int k = 0; k < _particles.Count; k++)
            if (_particles[k].Equals(b)) return (k < _density.Count) ? _density[k] : 0f;
        return 0f;
    }
    public float GetPressure(PhysicsBody b)
    {
        int i;
        if (_particleIndex.TryGetValue(b, out i) && i >= 0 && i < _pressure.Count) return _pressure[i];
        for (int k = 0; k < _particles.Count; k++)
            if (_particles[k].Equals(b)) return (k < _pressure.Count) ? _pressure[k] : 0f;
        return 0f;
    }
    public Vector2 GetSPHForce(PhysicsBody b)
    {
        int i;
        if (_particleIndex.TryGetValue(b, out i) && i >= 0 && i < _sphForce.Count) return _sphForce[i];
        for (int k = 0; k < _particles.Count; k++)
            if (_particles[k].Equals(b)) return (k < _sphForce.Count) ? _sphForce[k] : Vector2.zero;
        return Vector2.zero;
    }
    public Vector2 ComputeSPHForce(PhysicsBody body)
    {
        // Index-free: locate particle, ensure native snapshot covers it, run managed
        // passes over the native grid, return the matching force row.
        int i = -1;
        for (int k = 0; k < _particles.Count; k++)
            if (_particles[k].Equals(body)) { i = k; break; }
        if (i < 0) return Vector2.zero;
        if (_kernelDirty) BakeKernel();
        EnsureSphCapacity();
        RebuildGridFull();
        ComputeSPHDensitiesManaged(); ComputeSPHPressuresManaged(); ComputeSPHForcesManaged();
        return (i >= 0 && i < _sphForce.Count) ? _sphForce[i] : Vector2.zero;
    }
    public void ApplySPH(PhysicsBody body)
    {
        Vector2 f = ComputeSPHForce(body);
        if (f.sqrMagnitude >= 1e-10f) body.ApplyForceToCenter(f, true);
    }
    public void ApplySPHToAll() { StepPhysics(); }
    public void ComputeSPHDensities() { ComputeSPHDensitiesManaged(); }
    public void ComputeSPHPressures() { ComputeSPHPressuresManaged(); }
    public void ComputeSPHForces() { ComputeSPHForcesManaged(); }
    [System.Obsolete("Use smoothingRadius.")] public float repulsionRadius { get { return smoothingRadius; } set { smoothingRadius = value; } }
    [System.Obsolete("Use gasConstant.")] public float repulsionStrength { get { return gasConstant; } set { gasConstant = value; } }
    [System.Obsolete("Use enableSPH.")] public bool enableRepulsion { get { return enableSPH; } set { enableSPH = value; } }
    [System.Obsolete("Use ComputeSPHForce.")] public Vector2 ComputeRepulsionForce(PhysicsBody b) { return ComputeSPHForce(b); }
    [System.Obsolete("Use ApplySPH.")] public void ApplyRepulsion(PhysicsBody b) { ApplySPH(b); }
    [System.Obsolete("Use ApplySPHToAll.")] public void ApplyRepulsionToAll() { ApplySPHToAll(); }
    #endregion
    void DrawGridGizmo()
    {
        Gizmos.color = Color.cyan;
        Vector3 c = new Vector3(gridOrigin.x + gridSize.x * 0.5f, gridOrigin.y + gridSize.y * 0.5f, 0f);
        Gizmos.DrawWireCube(c, new Vector3(gridSize.x, gridSize.y, 0.01f));
        float cs = Mathf.Max(0.05f, cellSize);
        int nx = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cs));
        int ny = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cs));
        Gizmos.color = new Color(0f, 1f, 1f, 0.25f);
        for (int ix = 1; ix < nx; ix++)
        {
            float x = gridOrigin.x + ix * cs;
            Gizmos.DrawLine(new Vector3(x, gridOrigin.y, 0f), new Vector3(x, gridOrigin.y + gridSize.y, 0f));
        }
        for (int iy = 1; iy < ny; iy++)
        {
            float y = gridOrigin.y + iy * cs;
            Gizmos.DrawLine(new Vector3(gridOrigin.x, y, 0f), new Vector3(gridOrigin.x + gridSize.x, y, 0f));
        }
    }
    void OnDrawGizmos() { if (showGridGizmo && drawGridWhenUnselected) DrawGridGizmo(); }
    void OnDrawGizmosSelected() { if (showGridGizmo && !drawGridWhenUnselected) DrawGridGizmo(); }
}










