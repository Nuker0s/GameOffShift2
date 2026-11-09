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
    [Tooltip("Resolution: world size of one square cell.")]
    [Min(0.05f)] public float cellSize = 1f;

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
    [Tooltip("Smoothing radius h. Neighbours within h contribute. Keep ~3-4x particle radius.")]
    [Min(0.05f)] public float smoothingRadius = 0.4f;
    [Tooltip("SPH particle mass used ONLY when useBodyMass is OFF. When ON, body.mass (PhysicsCore) is authoritative.")]
    [Min(0.001f)] public float particleMass = 1f;
    [Tooltip("SPH rest density in (bodyMass * kernelWeight) units. NOT a PhysicsCore shape density. Pressure engages only above this value, so keep it near the calibrated hex-packing value (default matches mass=1, spacing=0.3, h=0.4). Recalibrate via context menu after curve/h changes.")]
    [Min(0.01f)] public float restDensity = 1.3f;
    [Tooltip("Pressure stiffness k. Pressure force also scales with kernel gradient, so steeper curve = stronger response at same k.")]
    public float gasConstant = 60f;
    [Tooltip("Viscosity coefficient mu, scaled by neighbour count (XSPH-style): effective damping per step ~ viscosity * neighbours / rho. Start 0.5-2. PhysicsCore linearDamping still applies on top in the engine.")]
    public float viscosity = 1f;
    [Tooltip("Maximum neighbours sampled per particle per step (0 = unlimited). Caps cost in dense piles; excess neighbours are skipped beyond this count.")]
    [Min(0)] public int maxNeighboursPerParticle = 32;
    [Tooltip("Normalize baked kernel so self-weight W(0) = 1. Density then reads as (mass-weighted effective neighbour count), making restDensity intuitive and curve-shape changes meaningful instead of rescaling everything.")]
    public bool normalizeKernel = true;
    [Tooltip("Clamp per-particle SPH force magnitude.")]
    [Min(0f)] public float maxForce = 200f;
    public bool enablePressure = true;
    public bool enableViscosity = true;
    [Tooltip("Allow p<0 when rho<rest (cohesive/tensile). OFF clamps p>=0 which strongly reduces permanent clumps.")]
    public bool allowNegativePressure = false;
    [Header("Anti-Clump (short-range overlap safety net — OFF by default)")]
    [Tooltip("Last-resort push for near-exact overlaps where the smooth kernel gradient -> 0. OFF by default: SPH pressure should do the work. Enable only if exact overlaps tunnel through pressure. If the sim is driven by this, restDensity/gasConstant are mistuned.")]
    public bool enableCoreRepulsion = false;
    [Tooltip("Core radius as fraction of h. Only pairs closer than this get the safety push.")]
    [Range(0.05f, 1f)] public float coreRadiusFraction = 0.2f;
    [Tooltip("Core stiffness scale. Actual push = stiffness * massPair * (1 - r/coreR), deliberately weaker than SPH pressure at normal densities.")]
    public float coreStiffness = 8f;
    [Tooltip("Guarantee a minimum repulsion slope for close pairs: effective |gradW| is at least minGradientFrac * (W0/h) for r < minGradientRange*h. Fixes the flat-centre problem where smooth-curve gradients -> 0 near contact and pressure goes mute. 0 = pure curve gradient.")]
    [Range(0f, 2f)] public float minCentralGradient = 0.5f;
    [Tooltip("Range (fraction of h) over which the minimum central gradient applies.")]
    [Range(0.05f, 1f)] public float minCentralRange = 0.45f;
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
    readonly List<PhysicsBody> _particles = new List<PhysicsBody>();
    readonly Dictionary<PhysicsBody, int> _particleCell = new Dictionary<PhysicsBody, int>();
    List<PhysicsBody>[] _cells;
    readonly List<float> _density = new List<float>();
    readonly List<float> _pressure = new List<float>();
    readonly List<Vector2> _sphForce = new List<Vector2>();
    readonly Dictionary<PhysicsBody, int> _particleIndex = new Dictionary<PhysicsBody, int>();
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
    int _nativeCapP = -1, _nativeCapC = -1;
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
        // PhysicsBody is a struct handle: (object)b is never null, so check validity instead.
        try { return b.isValid; } catch { return false; }
    }
    public Vector2 GetBodyVelocity(PhysicsBody b)
    {
        // PhysicsCore owns velocity. No fallback estimator: if the body is gone, return zero.
        if (!IsValidBody(b)) return Vector2.zero;
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
        _particleCell.Clear();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            if (!IsValidBody(b)) continue;
            int idx = GetCellIndexForPosition(b.position);
            _particleCell[b] = idx;
            if (idx >= 0) _cells[idx].Add(b);
        }
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
        int idx; if (_particleCell.TryGetValue(b, out idx)) return idx;
        return -1;
    }
    public int NeighbourCellSpan()
    {
        if (cellSize <= 0f) return 1;
        return Mathf.Max(1, Mathf.CeilToInt(smoothingRadius / cellSize));
    }
    void PruneNulls()
    {
        for (int i = _particles.Count - 1; i >= 0; i--)
        {
            if (!IsValidBody(_particles[i]))
            {
                try { _particleCell.Remove(_particles[i]); } catch { }
                _particles.RemoveAt(i);
            }
        }
    }
    void EnsureSphCapacity()
    {
        _particleIndex.Clear();
        for (int i = 0; i < _particles.Count; i++)
            if (IsValidBody(_particles[i])) _particleIndex[_particles[i]] = i;
        int n = _particles.Count;
        while (_density.Count < n) _density.Add(0f);
        while (_pressure.Count < n) _pressure.Add(0f);
        while (_sphForce.Count < n) _sphForce.Add(Vector2.zero);
        if (_density.Count > n) _density.RemoveRange(n, _density.Count - n);
        if (_pressure.Count > n) _pressure.RemoveRange(n, _pressure.Count - n);
        if (_sphForce.Count > n) _sphForce.RemoveRange(n, _sphForce.Count - n);
    }
    void EnsureNative(int pCount, int cCount)
    {
        if (pCount != _nativeCapP)
        {
            if (_posNative.IsCreated) _posNative.Dispose();
            if (_velNative.IsCreated) _velNative.Dispose();
            if (_massNative.IsCreated) _massNative.Dispose();
            if (_densNative.IsCreated) _densNative.Dispose();
            if (_pressNative.IsCreated) _pressNative.Dispose();
            if (_forceNative.IsCreated) _forceNative.Dispose();
            if (_entriesNative.IsCreated) _entriesNative.Dispose();
            _posNative = new NativeArray<float2>(Mathf.Max(1, pCount), Allocator.Persistent);
            _velNative = new NativeArray<float2>(Mathf.Max(1, pCount), Allocator.Persistent);
            _massNative = new NativeArray<float>(Mathf.Max(1, pCount), Allocator.Persistent);
            _densNative = new NativeArray<float>(Mathf.Max(1, pCount), Allocator.Persistent);
            _pressNative = new NativeArray<float>(Mathf.Max(1, pCount), Allocator.Persistent);
            _forceNative = new NativeArray<float2>(Mathf.Max(1, pCount), Allocator.Persistent);
            _entriesNative = new NativeArray<int>(Mathf.Max(1, pCount), Allocator.Persistent);
            _nativeCapP = pCount;
        }
        if (cCount != _nativeCapC)
        {
            if (_cellStartNative.IsCreated) _cellStartNative.Dispose();
            if (_cellEndNative.IsCreated) _cellEndNative.Dispose();
            _cellStartNative = new NativeArray<int>(Mathf.Max(1, cCount), Allocator.Persistent);
            _cellEndNative = new NativeArray<int>(Mathf.Max(1, cCount), Allocator.Persistent);
            _nativeCapC = cCount;
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
        if (_cellStartNative.IsCreated) _cellStartNative.Dispose();
        if (_cellEndNative.IsCreated) _cellEndNative.Dispose();
        if (_kernelNative.IsCreated) _kernelNative.Dispose();
        if (_kernelGradNative.IsCreated) _kernelGradNative.Dispose();
        _nativeCapP = _nativeCapC = -1;
    }
    void RebuildGridFull()
    {
        ClearCells();
        _particleCell.Clear();
        int n = _particles.Count;
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            int idx = GetCellIndexForPosition(b.position);
            _particleCell[b] = idx;
            if (idx >= 0) _cells[idx].Add(b);
        }
        int cells = _cellsX * _cellsY;
        EnsureNative(n, cells);
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            _posNative[i] = new float2(b.position.x, b.position.y);
            _velNative[i] = new float2(GetBodyVelocity(b).x, GetBodyVelocity(b).y);
            _massNative[i] = GetBodyMass(b);
        }
        int cursor = 0;
        for (int c = 0; c < cells; c++)
        {
            _cellStartNative[c] = cursor;
            List<PhysicsBody> list = _cells[c];
            for (int k = 0; k < list.Count; k++)
            {
                int pi; if (_particleIndex.TryGetValue(list[k], out pi)) _entriesNative[cursor++] = pi;
            }
            _cellEndNative[c] = cursor;
        }
    }
    #endregion
    [BurstCompile]
    struct DensityJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<float2> pos;
        [ReadOnly] public NativeArray<float> mass;
        [ReadOnly] public NativeArray<int> cellStart;
        [ReadOnly] public NativeArray<int> cellEnd;
        [ReadOnly] public NativeArray<int> entries;
        [ReadOnly] public NativeArray<float> kernel;
        public int cellsX, cellsY, span, lutN;
        public float2 origin, cellSz;
        public float h, h2, selfDensity;
        public NativeArray<float> density;
        public void Execute(int i)
        {
            float2 p = pos[i];
            float rho = selfDensity * mass[i];
            int cx = (int)math.floor((p.x - origin.x) / cellSz.x);
            int cy = (int)math.floor((p.y - origin.y) / cellSz.y);
            if (cx >= 0 && cy >= 0 && cx < cellsX && cy < cellsY)
            {
                for (int oy = -span; oy <= span; oy++)
                {
                    int ny = cy + oy; if (ny < 0 || ny >= cellsY) continue;
                    for (int ox = -span; ox <= span; ox++)
                    {
                        int nx = cx + ox; if (nx < 0 || nx >= cellsX) continue;
                        int c = ny * cellsX + nx;
                        for (int e = cellStart[c]; e < cellEnd[c]; e++)
                        {
                            int j = entries[e]; if (j == i) continue;
                            float2 d = p - pos[j];
                            float r2 = d.x * d.x + d.y * d.y;
                            if (r2 >= h2) continue;
                            float q = math.sqrt(math.max(r2, 0f)) / h;
                            float qq = math.clamp(q, 0f, 1f);
                            float ff = qq * (lutN - 1);
                            int k = (int)ff;
                            if (k < 0) k = 0; if (k >= lutN - 1) k = lutN - 2;
                            float tt = ff - k;
                            float w = kernel[k] * (1f - tt) + kernel[k + 1] * tt;
                            rho += mass[j] * w;
                        }
                    }
                }
            }
            density[i] = math.max(rho, 1e-5f);
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
        public float2 origin, cellSz;
        public float h, h2, visc, maxF, coreR, coreK;
        public float minGrad, minGradRange, w0;
        public int doP, doV, doCore, maxNeigh;
        public NativeArray<float2> force;
        public void Execute(int i)
        {
            float2 p = pos[i];
            float rhoA = math.max(density[i], 1e-5f);
            float pA = pressure[i];
            float2 f = float2.zero;
            int cx = (int)math.floor((p.x - origin.x) / cellSz.x);
            int cy = (int)math.floor((p.y - origin.y) / cellSz.y);
            if (cx >= 0 && cy >= 0 && cx < cellsX && cy < cellsY)
            {
                for (int oy = -span; oy <= span; oy++)
                {
                    int ny = cy + oy; if (ny < 0 || ny >= cellsY) continue;
                    for (int ox = -span; ox <= span; ox++)
                    {
                        int nx = cx + ox; if (nx < 0 || nx >= cellsX) continue;
                        int c = ny * cellsX + nx;
                        int countedN = 0;
                        for (int e = cellStart[c]; e < cellEnd[c]; e++)
                        {
                            int j = entries[e]; if (j == i) continue;
                            if (maxNeigh > 0 && countedN >= maxNeigh) break;
                            float2 delta = p - pos[j];
                            float r2 = delta.x * delta.x + delta.y * delta.y;
                            if (r2 >= h2) continue;
                            countedN++;
                            float mj = mass[j];
                            float mi = mass[i];
                            if (r2 < 1e-12f)
                            {
                                // Exactly overlapped: weak deterministic nudge along +X,
                                // mass-scaled like everything else. Safety net only.
                                if (doCore == 1 && coreK > 0f && coreR > 1e-6f)
                                    f += new float2(coreK * (mi + mj) * 0.5f, 0f);
                                continue;
                            }
                            float r = math.sqrt(r2);
                            float2 dir = delta / r;
                            float rhoB = math.max(density[j], 1e-5f);
                            float qq = math.clamp(r / h, 0f, 1f);
                            float ff = qq * (lutN - 1);
                            int k = (int)ff;
                            if (k < 0) k = 0; if (k >= lutN - 1) k = lutN - 2;
                            float tt = ff - k;
                            float gw = (kernelGrad[k] * (1f - tt) + kernelGrad[k + 1] * tt) / h;
                            // The smooth curve's gradient -> 0 near contact, muting pressure
                            // for close pairs. Enforce a minimum slope there so the curve
                            // SHAPE stays in charge: steepen the curve = stronger push.
                            if (minGrad > 0f && minGradRange > 1e-6f && r < minGradRange)
                            {
                                float need = -minGrad * (w0 / math.max(h, 1e-6f));
                                if (gw > need) gw = need;
                            }
                            // Symmetric SPH pressure: mass-weighted, stable under varied masses.
                            if (doP == 1) f -= mj * (pA / (rhoA * rhoA) + pressure[j] / (rhoB * rhoB)) * gw * dir;
                            if (doV == 1) f += mj * visc * (vel[j] - vel[i]) / rhoB * math.abs(gw);
                            // Safety net: weak mass-scaled push only deep inside coreR.
                            if (doCore == 1 && coreK > 0f && coreR > 1e-6f && r < coreR)
                                f += dir * (coreK * (mi + mj) * 0.5f * (1f - r / coreR));
                        }
                    }
                }
            }
            float l2 = f.x * f.x + f.y * f.y;
            if (maxF > 0f && l2 > maxF * maxF) f = f / math.sqrt(l2) * maxF;
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
        int cells = _cellsX * _cellsY;
        int span = NeighbourCellSpan();
        float h = Mathf.Max(0.05f, smoothingRadius);
        float selfW = _kernelLut != null && _kernelLut.Length > 0 ? _kernelLut[0] : 1f;
        var dj = new DensityJob
        {
            pos = _posNative, mass = _massNative,
            cellStart = _cellStartNative, cellEnd = _cellEndNative, entries = _entriesNative,
            kernel = _kernelNative, cellsX = _cellsX, cellsY = _cellsY, span = span,
            lutN = _kernelNative.Length,
            origin = new float2(gridOrigin.x, gridOrigin.y),
            cellSz = new float2(cellSize, cellSize),
            h = h, h2 = h * h, selfDensity = selfW, density = _densNative
        };
        dj.Schedule(n, Mathf.Max(1, jobBatchSize)).Complete();
        for (int i = 0; i < n; i++)
        {
            _density[i] = _densNative[i];
            _pressure[i] = ComputePressure(_densNative[i]);
            _pressNative[i] = _pressure[i];
        }
        float coreR = h * Mathf.Clamp01(coreRadiusFraction);
        var fj = new ForceJob
        {
            pos = _posNative, vel = _velNative, mass = _massNative,
            density = _densNative, pressure = _pressNative,
            cellStart = _cellStartNative, cellEnd = _cellEndNative, entries = _entriesNative,
            kernelGrad = _kernelGradNative, cellsX = _cellsX, cellsY = _cellsY, span = span,
            lutN = _kernelGradNative.Length,
            origin = new float2(gridOrigin.x, gridOrigin.y),
            cellSz = new float2(cellSize, cellSize),
            h = h, h2 = h * h, visc = viscosity, maxF = maxForce,
            coreR = coreR, coreK = coreStiffness,
            minGrad = minCentralGradient, minGradRange = minCentralRange * h,
            w0 = selfW,
            doP = enablePressure ? 1 : 0, doV = enableViscosity ? 1 : 0,
            doCore = enableCoreRepulsion ? 1 : 0, maxNeigh = maxNeighboursPerParticle,
            force = _forceNative
        };
        fj.Schedule(n, Mathf.Max(1, jobBatchSize)).Complete();
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            float2 f = _forceNative[i];
            Vector2 fv = new Vector2(f.x, f.y);
            _sphForce[i] = fv;
            // PhysicsCore integrates: we only add force at the centre (no torque).
            if (fv.sqrMagnitude >= 1e-10f) b.ApplyForceToCenter(fv, true);
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
        int n = _particles.Count;
        float h = Mathf.Max(0.05f, smoothingRadius);
        float selfW = _kernelLut != null && _kernelLut.Length > 0 ? _kernelLut[0] : 1f;
        for (int i = 0; i < n; i++)
        {
            PhysicsBody a = _particles[i];
            Vector2 pa = a.position;
            float mi = GetBodyMass(a);
            float rho = mi * selfW;
            int cx, cy;
            if (!WorldToCell(pa, out cx, out cy)) { _density[i] = Mathf.Max(rho, 1e-5f); continue; }
            int span = NeighbourCellSpan();
            for (int oy = -span; oy <= span; oy++)
            {
                int ny = cy + oy; if (ny < 0 || ny >= _cellsY) continue;
                for (int ox = -span; ox <= span; ox++)
                {
                    int nx = cx + ox; if (nx < 0 || nx >= _cellsX) continue;
                    var cell = _cells[ny * _cellsX + nx];
                    for (int k = 0; k < cell.Count; k++)
                    {
                        PhysicsBody b = cell[k];
                        if (!IsValidBody(b) || b.Equals(a)) continue;
                        float r = (pa - b.position).magnitude;
                        if (r >= h) continue;
                        rho += GetBodyMass(b) * SampleKernel(Mathf.Min(r, h * 0.999f) / h);
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
        float h = Mathf.Max(0.05f, smoothingRadius);
        float coreR = h * Mathf.Clamp01(coreRadiusFraction);
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody a = _particles[i];
            Vector2 pa = a.position;
            Vector2 va = GetBodyVelocity(a);
            float rhoA = Mathf.Max(_density[i], 1e-5f);
            float pA = _pressure[i];
            Vector2 f = Vector2.zero;
            int cx, cy;
            if (!WorldToCell(pa, out cx, out cy)) { _sphForce[i] = f; continue; }
            int span = NeighbourCellSpan();
            int countedN = 0;
            bool capHit = false;
            float w0m = (_kernelLut != null && _kernelLut.Length > 0) ? _kernelLut[0] : 1f;
            float minR = minCentralRange * h;
            for (int oy = -span; oy <= span && !capHit; oy++)
            {
                int ny = cy + oy; if (ny < 0 || ny >= _cellsY) continue;
                for (int ox = -span; ox <= span && !capHit; ox++)
                {
                    int nx = cx + ox; if (nx < 0 || nx >= _cellsX) continue;
                    var cell = _cells[ny * _cellsX + nx];
                    for (int k = 0; k < cell.Count; k++)
                    {
                        PhysicsBody b = cell[k];
                        if (!IsValidBody(b) || b.Equals(a)) continue;
                        if (maxNeighboursPerParticle > 0 && countedN >= maxNeighboursPerParticle) { capHit = true; break; }
                        Vector2 delta = pa - b.position;
                        float r = delta.magnitude;
                        if (r >= h) continue;
                        countedN++;
                        float mj = GetBodyMass(b);
                        float mi = GetBodyMass(a);
                        if (r < 1e-6f)
                        {
                            // Exactly overlapped: weak mass-scaled nudge along +X. Safety net only.
                            if (enableCoreRepulsion) f.x += coreStiffness * (mi + mj) * 0.5f;
                            continue;
                        }
                        Vector2 dir = delta / r;
                        int j; _particleIndex.TryGetValue(b, out j);
                        float rhoB = (j >= 0 && j < _density.Count) ? Mathf.Max(_density[j], 1e-5f) : rhoA;
                        float pB = (j >= 0 && j < _pressure.Count) ? _pressure[j] : pA;
                        float gw = SampleKernelGradient(r / h) / h;
                        // Minimum central slope: keeps pressure alive for close pairs
                        // where the smooth curve flattens, so curve steepness stays in charge.
                        if (minCentralGradient > 0f && minR > 1e-6f && r < minR && r > 1e-6f)
                        {
                            float need = -minCentralGradient * (w0m / Mathf.Max(h, 1e-6f));
                            if (gw > need) gw = need;
                        }
                        // Symmetric form: mj * (pA/rhoA^2 + pB/rhoB^2), mass-aware.
                        if (enablePressure) f -= mj * (pA / (rhoA * rhoA) + pB / (rhoB * rhoB)) * gw * dir;
                        // No manual damping compensation: PhysicsCore owns linearDamping.
                        if (enableViscosity) f += mj * viscosity * (GetBodyVelocity(b) - va) / rhoB * Mathf.Abs(gw);
                        // Safety net only: weak mass-scaled push deep inside coreR.
                        if (enableCoreRepulsion && coreR > 1e-6f && r < coreR)
                            f += dir * (coreStiffness * (mi + mj) * 0.5f * (1f - r / coreR));
                    }
                }
            }
            if (maxForce > 0f && f.sqrMagnitude > maxForce * maxForce) f = f.normalized * maxForce;
            _sphForce[i] = f;
        }
    }
    #region Public API (kept compatible)
    public bool Contains(PhysicsBody body)
    {
        if (!IsValidBody(body)) return false;
        return _particleCell.ContainsKey(body);
    }
    public bool AddParticle(PhysicsBody body)
    {
        if (!IsValidBody(body) || _particleCell.ContainsKey(body)) return false;
        if (_cells == null) BuildGrid();
        _particles.Add(body);
        int idx = GetCellIndexForPosition(body.position);
        _particleCell[body] = idx;
        if (idx >= 0) _cells[idx].Add(body);
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
        int idx; if (!_particleCell.TryGetValue(body, out idx)) return false;
        if (idx >= 0 && _cells != null && idx < _cells.Length) _cells[idx].Remove(body);
        _particleCell.Remove(body); _particles.Remove(body);
        _particleIndex.Clear();
        return true;
    }
    public void ClearParticles()
    {
        _particles.Clear(); _particleCell.Clear(); _particleIndex.Clear();
        _density.Clear(); _pressure.Clear(); _sphForce.Clear();
        ClearCells();
    }
    public void SyncFromBenchmark(BenchMark source, bool removeMissing = false)
    {
        if (source == null || source.Bodies == null) return;
        var src = source.Bodies;
        for (int i = 0; i < src.Count; i++)
            if (IsValidBody(src[i]) && !_particleCell.ContainsKey(src[i])) AddParticle(src[i]);
        if (removeMissing)
        {
            var set = new HashSet<PhysicsBody>(src);
            for (int i = _particles.Count - 1; i >= 0; i--)
                if (!set.Contains(_particles[i])) RemoveParticle(_particles[i]);
        }
    }
    public bool UpdateParticleCell(PhysicsBody body)
    {
        if (!IsValidBody(body) || _cells == null) return false;
        int oldIdx; if (!_particleCell.TryGetValue(body, out oldIdx)) return false;
        int ni = GetCellIndexForPosition(body.position);
        if (ni == oldIdx) return false;
        if (oldIdx >= 0 && oldIdx < _cells.Length) _cells[oldIdx].Remove(body);
        if (ni >= 0 && ni < _cells.Length) _cells[ni].Add(body);
        _particleCell[body] = ni;
        return true;
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
        if (!IsValidBody(body) || _cells == null) return;
        int cx, cy; if (!WorldToCell(body.position, out cx, out cy)) return;
        int span = radius <= 0f ? 1 : Mathf.Max(1, Mathf.CeilToInt(radius / Mathf.Max(0.0001f, cellSize)));
        float r2 = radius * radius; Vector2 p = body.position;
        for (int oy = -span; oy <= span; oy++)
        {
            int ny = cy + oy; if (ny < 0 || ny >= _cellsY) continue;
            for (int ox = -span; ox <= span; ox++)
            {
                int nx = cx + ox; if (nx < 0 || nx >= _cellsX) continue;
                var cell = _cells[ny * _cellsX + nx];
                for (int i = 0; i < cell.Count; i++)
                {
                    var o = cell[i];
                    if (!IsValidBody(o) || o.Equals(body)) continue;
                    if (radius > 0f && (o.position - p).sqrMagnitude >= r2) continue;
                    outN.Add(o);
                }
            }
        }
    }
    public void GetNeighbours3x3(PhysicsBody body, List<PhysicsBody> outN) { GetNeighbours(body, outN, cellSize * 1.5f); }
    public List<PhysicsBody> GetCellContents(int cx, int cy)
    {
        if (_cells == null || !IsValidCell(cx, cy)) return null;
        return _cells[CellToIndex(cx, cy)];
    }
    public float GetDensity(PhysicsBody b)
    {
        int i; if (!_particleIndex.TryGetValue(b, out i)) return 0f;
        return (i >= 0 && i < _density.Count) ? _density[i] : 0f;
    }
    public float GetPressure(PhysicsBody b)
    {
        int i; if (!_particleIndex.TryGetValue(b, out i)) return 0f;
        return (i >= 0 && i < _pressure.Count) ? _pressure[i] : 0f;
    }
    public Vector2 GetSPHForce(PhysicsBody b)
    {
        int i; if (!_particleIndex.TryGetValue(b, out i)) return Vector2.zero;
        return (i >= 0 && i < _sphForce.Count) ? _sphForce[i] : Vector2.zero;
    }
    public Vector2 ComputeSPHForce(PhysicsBody body)
    {
        int i; if (!_particleIndex.TryGetValue(body, out i)) return Vector2.zero;
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










