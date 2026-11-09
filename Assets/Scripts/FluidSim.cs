using System.Collections.Generic;
using System.Reflection;
using UnityEngine;
using Unity.U2D.Physics;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

/// <summary>
/// FluidSim - Burst + Jobs SPH over a uniform grid, PhysicsCore2D-driven particles.
///
/// GRID DECISION (you asked which is optimal):
///   We REBUILD the grid every step (full rebuild), particles do NOT maintain
///   their own cells incrementally. Reason: incremental List.Remove/Add per move
///   is O(occupants) + GC churn + branchy, and cannot be Bursted. A full rebuild
///   is O(N + cells) counting-sort, cache friendly, alloc-free after warmup, and
///   feeds Burst jobs directly (cellStart/cellEnd + sorted entries). This file
///   keeps UpdateParticleCell() only for external queries; the hot path always
///   calls RebuildGridFull().
/// BENCHMARK AUDIT (Benchmark.cs properties vs simulation):
///   - linearDamping=0.2  -> extra velocity damping on top of SPH viscosity.
///     Visually thicker fluid. OK, or set 0 and let viscosity do it. See
///     compensateLinearDamping toggle.
///   - gravityScale=1      -> fine, SPH forces add on top of gravity.
///   - density=1 (shape)   -> sets rigidbody mass. If useBodyMass=true we read
///     it as particle mass (good: per-type mass via PhysicsCore, no FluidSim logic).
///   - friction=0.2 / bounciness=0.2 -> only affect collision contacts, NOT SPH.
///     No interference unless particles collide with walls.
///   - radius=0.10          -> keep smoothingRadius ~= 3-4x radius (0.3-0.4).
///     Your default 1.0 is very blobby; try 0.4.
///   - triggerEvents=true  -> only enables callbacks, does not disable SPH.
/// PARTICLE TYPES (no FluidSim special-casing, as requested):
///   Types come from PhysicsCore2D state, never from an if(type==X) in here.
///   - Mass: useBodyMass reads body mass (from shape density) per particle.
///   - Velocity: useBodyVelocity reads body linearVelocity per particle.
///   - Type id for rendering: GetParticleTypeKey(body) quantizes gravityScale
///     + linearDamping into a float key BenchMark can push to the shader (z).
///     Define new fluids by changing those PhysicsCore values, not this file.
/// KERNEL CURVE: smoothingCurve (0=center,1=edge) is baked to a 64-entry LUT.
///   Default is a Poly6-ish smooth falloff. Edit it in the Inspector for
///   goopier/sharper fluids. Density/force jobs only sample the LUT, so any
///   curve is Burst-compatible.
/// </summary>
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
    [Tooltip("Kernel weight vs normalized distance q=r/h. q=0 centre, q=1 edge. Should fall to 0 at q=1.")]
    public AnimationCurve smoothingCurve = new AnimationCurve(
        new Keyframe(0f, 1f, 0f, 0f),
        new Keyframe(0.35f, 0.72f, -1.2f, -1.2f),
        new Keyframe(0.7f, 0.28f, -1.4f, -1.4f),
        new Keyframe(1f, 0f, -0.8f, 0f));
    [Tooltip("LUT entries baked from the curve. Higher = smoother gradients, slightly slower bake.")]
    [Range(16, 256)] public int kernelLutSize = 64;
    [Tooltip("Re-bake LUT automatically when curve edits in Inspector.")]
    public bool autoRebakeKernel = true;
    [Header("SPH")]
    [Tooltip("Smoothing radius h. Neighbours within h contribute.")]
    [Min(0.05f)] public float smoothingRadius = 0.4f;
    [Tooltip("Fallback mass when useBodyMass finds no readable PhysicsCore mass.")]
    [Min(0.001f)] public float particleMass = 1f;
    [Tooltip("Rest density. Pressure = stiffness * (density - restDensity).")]
    [Min(0.01f)] public float restDensity = 4f;
    [Tooltip("Pressure stiffness k.")]
    public float gasConstant = 20f;
    [Tooltip("Viscosity coefficient mu.")]
    public float viscosity = 1.5f;
    [Tooltip("Clamp per-particle SPH force magnitude.")]
    [Min(0f)] public float maxForce = 200f;
    public bool enablePressure = true;
    public bool enableViscosity = true;
    public bool enableSPH = true;
    public bool runInFixedUpdate = true;

    [Header("PhysicsCore2D particle properties (no custom types)")]
    [Tooltip("Read per-particle mass from PhysicsCore body mass when available.")]
    public bool useBodyMass = true;
    [Tooltip("Read per-particle velocity from PhysicsCore linear velocity when available.")]
    public bool useBodyVelocity = true;
    [Tooltip("How a float type key is derived from PhysicsCore props for rendering/grouping.")]
    public ParticleTypeSource particleTypeSource = ParticleTypeSource.GravityPlusDamping;
    [Tooltip("Quantization step for the derived type key.")]
    [Min(0.001f)] public float typeQuantize = 0.05f;
    [Tooltip("Divide SPH viscosity by (1+damping*dt) so Benchmark linearDamping does not double-damp. 0 keeps raw.")]
    public bool compensateLinearDamping = true;
    [Tooltip("Fallback velocityDamping used when body exposes no damping property. Mirrors Benchmark default 0.2.")]
    public float fallbackDamping = 0.2f;

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
    readonly Dictionary<PhysicsBody, Vector2> _prevPos = new Dictionary<PhysicsBody, Vector2>();
    float _fixedDt = 0.02f;
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
    static PhysPropProbe s_massProbe = new PhysPropProbe(
        new[] { "mass", "Mass", "bodyMass", "BodyMass", "m_mass" });
    static PhysPropProbe s_velProbe = new PhysPropProbe(
        new[] { "linearVelocity", "LinearVelocity", "velocity", "Velocity", "linear_velocity" });
    static PhysPropProbe s_dampProbe = new PhysPropProbe(
        new[] { "linearDamping", "LinearDamping", "damping", "Damping", "linear_damping" });
    static PhysPropProbe s_gravProbe = new PhysPropProbe(
        new[] { "gravityScale", "GravityScale", "gravity_scale" });
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
        kernelLutSize = Mathf.Clamp(kernelLutSize, 16, 256);
        jobBatchSize = Mathf.Max(1, jobBatchSize);
        _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cellSize));
        _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cellSize));
        if (autoRebakeKernel) _kernelDirty = true;
        if (Application.isPlaying)
        {
            if (_kernelDirty) BakeKernel();
            BuildGrid();
        }
    }

    void FixedUpdate() { if (runInFixedUpdate) StepPhysics(); }

    public void StepPhysics()
    {
        if (_cells == null) BuildGrid();
        if (autoSyncSource != null) SyncFromBenchmark(autoSyncSource, autoSyncRemoveMissing);
        if (_kernelDirty) BakeKernel();
        if (!enableSPH || _particles.Count == 0) return;
        _fixedDt = Application.isPlaying ? Mathf.Max(Time.fixedDeltaTime, 1e-5f) : 0.02f;
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
            smoothingCurve = AnimationCurve.EaseInOut(0f, 1f, 1f, 0f);
        for (int i = 0; i < n; i++)
        {
            float q = (float)i / (n - 1);
            _kernelLut[i] = Mathf.Max(0f, smoothingCurve.Evaluate(q));
        }
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
    sealed class PhysPropProbe
    {
        readonly string[] _names;
        PropertyInfo[] _props; FieldInfo[] _fields; bool _resolved;
        public PhysPropProbe(string[] names) { _names = names; }
        void Resolve(System.Type t)
        {
            _props = new PropertyInfo[_names.Length];
            _fields = new FieldInfo[_names.Length];
            for (int i = 0; i < _names.Length; i++)
            {
                _props[i] = t.GetProperty(_names[i], BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                if (_props[i] == null)
                    _fields[i] = t.GetField(_names[i], BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
            }
            _resolved = true;
        }
        public bool TryGetFloat(object o, out float v)
        {
            v = 0f; if (o == null) return false;
            if (!_resolved) Resolve(o.GetType());
            for (int i = 0; i < _names.Length; i++)
            {
                try
                {
                    object val = _props[i] != null ? _props[i].GetValue(o, null)
                        : _fields[i] != null ? _fields[i].GetValue(o) : null;
                    if (val is float f) { v = f; return true; }
                    if (val is double d) { v = (float)d; return true; }
                    if (val is int ii) { v = ii; return true; }
                }
                catch { }
            }
            return false;
        }
        public bool TryGetVec(object o, out Vector2 v)
        {
            v = Vector2.zero; if (o == null) return false;
            if (!_resolved) Resolve(o.GetType());
            for (int i = 0; i < _names.Length; i++)
            {
                try
                {
                    object val = _props[i] != null ? _props[i].GetValue(o, null)
                        : _fields[i] != null ? _fields[i].GetValue(o) : null;
                    if (val is Vector2 vv) { v = vv; return true; }
                    if (val is float2 f2) { v = new Vector2(f2.x, f2.y); return true; }
                }
                catch { }
            }
            return false;
        }
    }

    public float GetBodyMass(PhysicsBody b)
    {
        float v;
        if (useBodyMass && (object)b != null && s_massProbe.TryGetFloat(b, out v) && v > 1e-6f && v < 1e6f) return v;
        return particleMass;
    }
    Vector2 EstimateVelocityFallback(PhysicsBody b)
    {
        Vector2 p; if (!_prevPos.TryGetValue(b, out p)) return Vector2.zero;
        if (_fixedDt <= 1e-6f) return Vector2.zero;
        return (b.position - p) / _fixedDt;
    }
    public Vector2 GetBodyVelocity(PhysicsBody b)
    {
        Vector2 fv = EstimateVelocityFallback(b);
        if (!useBodyVelocity || (object)b == null) return fv;
        Vector2 v; if (s_velProbe.TryGetVec(b, out v) && v.sqrMagnitude < 1e7f) return v;
        return fv;
    }
    public float GetBodyDamping(PhysicsBody b)
    {
        float v; if (s_dampProbe.TryGetFloat(b, out v)) return Mathf.Max(0f, v);
        return fallbackDamping;
    }
    public float GetBodyGravity(PhysicsBody b)
    {
        float v; if (s_gravProbe.TryGetFloat(b, out v)) return v;
        return 1f;
    }
    public float GetParticleTypeKey(PhysicsBody b)
    {
        if ((object)b == null) return 0f;
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
            if ((object)b == null) continue;
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
            if ((object)_particles[i] == null)
            {
                try { _particleCell.Remove(_particles[i]); } catch { }
                try { _prevPos.Remove(_particles[i]); } catch { }
                _particles.RemoveAt(i);
            }
        }
    }
    void EnsureSphCapacity()
    {
        _particleIndex.Clear();
        for (int i = 0; i < _particles.Count; i++)
            if ((object)_particles[i] != null) _particleIndex[_particles[i]] = i;
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
                            if (r2 >= h2 || r2 < 1e-12f) continue;
                            float q = math.sqrt(r2) / h;
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
        [ReadOnly] public NativeArray<float> density;
        [ReadOnly] public NativeArray<float> pressure;
        [ReadOnly] public NativeArray<int> cellStart;
        [ReadOnly] public NativeArray<int> cellEnd;
        [ReadOnly] public NativeArray<int> entries;
        [ReadOnly] public NativeArray<float> kernelGrad;
        public int cellsX, cellsY, span, lutN;
        public float2 origin, cellSz;
        public float h, h2, visc, maxF;
        public int doP, doV;
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
                        for (int e = cellStart[c]; e < cellEnd[c]; e++)
                        {
                            int j = entries[e]; if (j == i) continue;
                            float2 delta = p - pos[j];
                            float r2 = delta.x * delta.x + delta.y * delta.y;
                            if (r2 >= h2 || r2 < 1e-12f) continue;
                            float r = math.sqrt(r2);
                            float2 dir = delta / r;
                            float rhoB = math.max(density[j], 1e-5f);
                            float qq = math.clamp(r / h, 0f, 1f);
                            float ff = qq * (lutN - 1);
                            int k = (int)ff;
                            if (k < 0) k = 0; if (k >= lutN - 1) k = lutN - 2;
                            float tt = ff - k;
                            float gw = (kernelGrad[k] * (1f - tt) + kernelGrad[k + 1] * tt) / h;
                            if (doP == 1) f -= ((pA + pressure[j]) / (2f * rhoB)) * gw * dir;
                            if (doV == 1) f += visc * (vel[j] - vel[i]) / rhoB * math.abs(gw);
                        }
                    }
                }
            }
            float l2 = f.x * f.x + f.y * f.y;
            if (maxF > 0f && l2 > maxF * maxF) f = f / math.sqrt(l2) * maxF;
            force[i] = f;
        }
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
            _pressure[i] = Mathf.Max(0f, gasConstant * (_densNative[i] - restDensity));
            _pressNative[i] = _pressure[i];
        }
        var fj = new ForceJob
        {
            pos = _posNative, vel = _velNative, density = _densNative, pressure = _pressNative,
            cellStart = _cellStartNative, cellEnd = _cellEndNative, entries = _entriesNative,
            kernelGrad = _kernelGradNative, cellsX = _cellsX, cellsY = _cellsY, span = span,
            lutN = _kernelGradNative.Length,
            origin = new float2(gridOrigin.x, gridOrigin.y),
            cellSz = new float2(cellSize, cellSize),
            h = h, h2 = h * h, visc = viscosity, maxF = maxForce,
            doP = enablePressure ? 1 : 0, doV = enableViscosity ? 1 : 0, force = _forceNative
        };
        fj.Schedule(n, Mathf.Max(1, jobBatchSize)).Complete();
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b = _particles[i];
            float2 f = _forceNative[i];
            Vector2 fv = new Vector2(f.x, f.y);
            _sphForce[i] = fv;
            if (fv.sqrMagnitude >= 1e-10f) b.ApplyForce(fv, b.position);
            _prevPos[b] = b.position;
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
            if (f.sqrMagnitude >= 1e-10f) b.ApplyForce(f, b.position);
            _prevPos[b] = b.position;
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
                        if ((object)b == null || b.Equals(a)) continue;
                        float r = (pa - b.position).magnitude;
                        if (r >= h || r < 1e-6f) continue;
                        rho += GetBodyMass(b) * SampleKernel(r / h);
                    }
                }
            }
            _density[i] = Mathf.Max(rho, 1e-5f);
        }
    }
    public void ComputeSPHPressuresManaged()
    {
        for (int i = 0; i < _particles.Count; i++)
            _pressure[i] = gasConstant * (_density[i] - restDensity);
    }
    public void ComputeSPHForcesManaged()
    {
        float h = Mathf.Max(0.05f, smoothingRadius);
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody a = _particles[i];
            Vector2 pa = a.position;
            Vector2 va = GetBodyVelocity(a);
            float rhoA = Mathf.Max(_density[i], 1e-5f);
            Vector2 f = Vector2.zero;
            int cx, cy;
            if (!WorldToCell(pa, out cx, out cy)) { _sphForce[i] = f; continue; }
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
                        if ((object)b == null || b.Equals(a)) continue;
                        Vector2 delta = pa - b.position;
                        float r = delta.magnitude;
                        if (r >= h || r < 1e-6f) continue;
                        Vector2 dir = delta / r;
                        int j; _particleIndex.TryGetValue(b, out j);
                        float rhoB = (j >= 0 && j < _density.Count) ? Mathf.Max(_density[j], 1e-5f) : rhoA;
                        float pB = (j >= 0 && j < _pressure.Count) ? _pressure[j] : _pressure[i];
                        float gw = SampleKernelGradient(r / h) / h;
                        if (enablePressure) f -= ((_pressure[i] + pB) / (2f * rhoB)) * gw * dir;
                        if (enableViscosity)
                        {
                            Vector2 vb = GetBodyVelocity(b);
                            float comp = 1f;
                            if (compensateLinearDamping) comp = 1f + GetBodyDamping(b) * _fixedDt;
                            f += (viscosity / comp) * (vb - va) / rhoB * Mathf.Abs(gw);
                        }
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
        if ((object)body == null) return false;
        return _particleCell.ContainsKey(body);
    }
    public bool AddParticle(PhysicsBody body)
    {
        if ((object)body == null || _particleCell.ContainsKey(body)) return false;
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
        if ((object)body == null) return false;
        int idx; if (!_particleCell.TryGetValue(body, out idx)) return false;
        if (idx >= 0 && _cells != null && idx < _cells.Length) _cells[idx].Remove(body);
        _particleCell.Remove(body); _particles.Remove(body);
        _prevPos.Remove(body); _particleIndex.Clear();
        return true;
    }
    public void ClearParticles()
    {
        _particles.Clear(); _particleCell.Clear(); _prevPos.Clear(); _particleIndex.Clear();
        _density.Clear(); _pressure.Clear(); _sphForce.Clear();
        ClearCells();
    }
    public void SyncFromBenchmark(BenchMark source, bool removeMissing = false)
    {
        if (source == null || source.Bodies == null) return;
        var src = source.Bodies;
        for (int i = 0; i < src.Count; i++)
            if ((object)src[i] != null && !_particleCell.ContainsKey(src[i])) AddParticle(src[i]);
        if (removeMissing)
        {
            var set = new HashSet<PhysicsBody>(src);
            for (int i = _particles.Count - 1; i >= 0; i--)
                if (!set.Contains(_particles[i])) RemoveParticle(_particles[i]);
        }
    }
    public bool UpdateParticleCell(PhysicsBody body)
    {
        if ((object)body == null || _cells == null) return false;
        int oldIdx; if (!_particleCell.TryGetValue(body, out oldIdx)) return false;
        int ni = GetCellIndexForPosition(body.position);
        if (ni == oldIdx) return false;
        if (oldIdx >= 0 && oldIdx < _cells.Length) _cells[oldIdx].Remove(body);
        if (ni >= 0 && ni < _cells.Length) _cells[ni].Add(body);
        _particleCell[body] = ni;
        return true;
    }
    public void UpdateAllParticleCells() { PruneNulls(); EnsureSphCapacity(); RebuildGridFull(); }
    public void GetNeighbours(PhysicsBody body, List<PhysicsBody> outN, float radius)
    {
        outN.Clear();
        if ((object)body == null || _cells == null) return;
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
                    if ((object)o == null || o.Equals(body)) continue;
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
        if (f.sqrMagnitude >= 1e-10f) body.ApplyForce(f, body.position);
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










