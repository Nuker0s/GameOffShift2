using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// FluidSim - uniform spatial hash grid for 2D physics particles + simplified SPH.
/// Grid covers [gridOrigin, gridOrigin + gridSize] in square cells of cellSize.
/// Particles re-compute their cell every step by flooring their position.
/// SPH pass (Muller et al. 2003, simplified): Poly6 density, Spiky pressure gradient,
/// viscosity Laplacian. Neighbour search spans ceil(smoothingRadius / cellSize) cells
/// so h may exceed cellSize. Plain C# loops on purpose; see bottom notes for Burst mapping.
/// </summary>
public class FluidSim : MonoBehaviour
{
    [Header("Grid Definition (predefined size + resolution)")]
    [Tooltip("World-space bottom-left corner of the grid.")]
    public Vector2 gridOrigin = new Vector2(-10f, -10f);
    [Tooltip("World-space size of the grid (width, height).")]
    public Vector2 gridSize = new Vector2(20f, 20f);
    [Tooltip("Resolution: world size of one square cell.")]
    [Min(0.05f)]
    public float cellSize = 1f;

    [Header("SPH (replaces simple repulsion)")]
    [Tooltip("Smoothing radius h. Neighbours within h contribute. Search spans ceil(h / cellSize) cells.")]
    [Min(0.05f)]
    public float smoothingRadius = 1f;
    [Tooltip("Assumed mass per particle.")]
    [Min(0.001f)]
    public float particleMass = 1f;
    [Tooltip("Rest density. Pressure = gasConstant * (density - restDensity).")]
    [Min(0.01f)]
    public float restDensity = 2f;
    [Tooltip("Gas constant k: stiffness of pressure response.")]
    public float gasConstant = 20f;
    [Tooltip("Viscosity coefficient mu: damps relative velocity.")]
    public float viscosity = 1.5f;
    [Tooltip("Clamp per-particle SPH force magnitude for stability.")]
    [Min(0f)]
    public float maxForce = 200f;
    public bool enablePressure = true;
    public bool enableViscosity = true;
    public bool enableSPH = true;
    public bool runInFixedUpdate = true;

    [Header("Optional Auto-Sync")]
    [Tooltip("If assigned, bodies in this BenchMark are auto-registered each physics step.")]
    public BenchMark autoSyncSource;
    public bool autoSyncRemoveMissing = false;

    int _cellsX = 1;
    int _cellsY = 1;
    List<PhysicsBody>[] _cells;
    readonly List<PhysicsBody> _particles = new List<PhysicsBody>();
    readonly Dictionary<PhysicsBody, int> _particleCell = new Dictionary<PhysicsBody, int>();

    // SPH per-step caches (indexed parallel to _particles).
    readonly List<float> _density = new List<float>();
    readonly List<float> _pressure = new List<float>();
    readonly List<Vector2> _sphForce = new List<Vector2>();
    readonly Dictionary<PhysicsBody, Vector2> _prevPos = new Dictionary<PhysicsBody, Vector2>();
    readonly Dictionary<PhysicsBody, int> _particleIndex = new Dictionary<PhysicsBody, int>();
    float _fixedDt = 0.02f;

    public int CellsX => _cellsX;
    public int CellsY => _cellsY;
    public int ParticleCount => _particles.Count;
    public IReadOnlyList<PhysicsBody> Particles => _particles;

    void Awake() { BuildGrid(); }

    void OnValidate()
    {
        cellSize = Mathf.Max(0.05f, cellSize);
        gridSize.x = Mathf.Max(cellSize, gridSize.x);
        gridSize.y = Mathf.Max(cellSize, gridSize.y);
        smoothingRadius = Mathf.Max(0.05f, smoothingRadius);
        particleMass = Mathf.Max(0.001f, particleMass);
        restDensity = Mathf.Max(0.01f, restDensity);
        if (Application.isPlaying && _cells != null) BuildGrid();
        else
        {
            _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cellSize));
            _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cellSize));
        }
    }

    void FixedUpdate()
    {
        if (runInFixedUpdate) StepPhysics();
    }

    public void StepPhysics()
    {
        if (_cells == null) BuildGrid();
        if (autoSyncSource != null) SyncFromBenchmark(autoSyncSource, autoSyncRemoveMissing);
        UpdateAllParticleCells();
        if (enableSPH) ApplySPHToAll();
    }

    #region Grid construction
    public void BuildGrid()
    {
        _cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cellSize));
        _cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cellSize));
        _cells = new List<PhysicsBody>[_cellsX * _cellsY];
        for (int i = 0; i < _cells.Length; i++)
            _cells[i] = new List<PhysicsBody>(8);
        _particleCell.Clear();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            if ((object)b == null) continue;
            int idx = GetCellIndexForPosition(b.position);
            _particleCell[b] = idx;
            if (idx >= 0) _cells[idx].Add(b);
        }
        // Drop stale velocity history for bodies no longer registered.
        if (_prevPos.Count > 0)
        {
            var stale = new List<PhysicsBody>();
            foreach (var kv in _prevPos)
                if (!_particleCell.ContainsKey(kv.Key)) stale.Add(kv.Key);
            for (int i = 0; i < stale.Count; i++) _prevPos.Remove(stale[i]);
        }
    }

    public void ClearCells()
    {
        if (_cells == null) return;
        for (int i = 0; i < _cells.Length; i++) _cells[i].Clear();
    }
    #endregion

    #region Coordinate helpers (flooring)
    public bool WorldToCell(Vector2 worldPos, out int cx, out int cy)
    {
        cx = Mathf.FloorToInt((worldPos.x - gridOrigin.x) / cellSize);
        cy = Mathf.FloorToInt((worldPos.y - gridOrigin.y) / cellSize);
        return cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY;
    }

    public int CellToIndex(int cx, int cy) { return cy * _cellsX + cx; }
    public bool IsValidCell(int cx, int cy) { return cx >= 0 && cy >= 0 && cx < _cellsX && cy < _cellsY; }

    public int GetCellIndexForPosition(Vector2 worldPos)
    {
        int cx, cy;
        if (!WorldToCell(worldPos, out cx, out cy)) return -1;
        return CellToIndex(cx, cy);
    }

    public int GetParticleCellIndex(PhysicsBody body)
    {
        int idx;
        if (_particleCell.TryGetValue(body, out idx)) return idx;
        return -1;
    }
    #endregion

    #region Add / Remove
    public bool Contains(PhysicsBody body)
    {
        if ((object)body == null) return false;
        return _particleCell.ContainsKey(body);
    }

    public bool AddParticle(PhysicsBody body)
    {
        if ((object)body == null) return false;
        if (_particleCell.ContainsKey(body)) return false;
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
        foreach (var b in bodies) { if (AddParticle(b)) added++; }
        return added;
    }

    public bool RemoveParticle(PhysicsBody body)
    {
        if ((object)body == null) return false;
        int idx;
        if (!_particleCell.TryGetValue(body, out idx)) return false;
        if (idx >= 0 && _cells != null && idx < _cells.Length) _cells[idx].Remove(body);
        _particleCell.Remove(body);
        _particles.Remove(body);
        _prevPos.Remove(body);
        _particleIndex.Clear();
        return true;
    }

    public void ClearParticles()
    {
        _particles.Clear();
        _particleCell.Clear();
        _prevPos.Clear();
        _particleIndex.Clear();
        _density.Clear();
        _pressure.Clear();
        _sphForce.Clear();
        ClearCells();
    }

    public void SyncFromBenchmark(BenchMark source, bool removeMissing = false)
    {
        if (source == null || source.Bodies == null) return;
        var srcBodies = source.Bodies;
        for (int i = 0; i < srcBodies.Count; i++)
        {
            PhysicsBody b = srcBodies[i];
            if ((object)b == null) continue;
            if (!_particleCell.ContainsKey(b)) AddParticle(b);
        }
        if (removeMissing)
        {
            var srcSet = new HashSet<PhysicsBody>(srcBodies);
            for (int i = _particles.Count - 1; i >= 0; i--)
                if (!srcSet.Contains(_particles[i])) RemoveParticle(_particles[i]);
        }
    }
    #endregion

    #region Automatic cell movement
    public bool UpdateParticleCell(PhysicsBody body)
    {
        if ((object)body == null || _cells == null) return false;
        int oldIdx;
        if (!_particleCell.TryGetValue(body, out oldIdx)) return false;
        int newIdx = GetCellIndexForPosition(body.position);
        if (newIdx == oldIdx) return false;
        if (oldIdx >= 0 && oldIdx < _cells.Length) _cells[oldIdx].Remove(body);
        if (newIdx >= 0 && newIdx < _cells.Length) _cells[newIdx].Add(body);
        _particleCell[body] = newIdx;
        return true;
    }

    public void UpdateAllParticleCells()
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
        ClearCells();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            int idx = GetCellIndexForPosition(b.position);
            _particleCell[b] = idx;
            if (idx >= 0) _cells[idx].Add(b);
        }
    }
    #endregion

    #region Neighbour queries + SPH
    /// <summary>Cell span needed to cover smoothingRadius: span = ceil(h / cellSize).</summary>
    public int NeighbourCellSpan()
    {
        if (cellSize <= 0f) return 1;
        return Mathf.Max(1, Mathf.CeilToInt(smoothingRadius / cellSize));
    }

    public void GetNeighbours(PhysicsBody body, List<PhysicsBody> outNeighbours, float radius)
    {
        outNeighbours.Clear();
        if ((object)body == null || _cells == null) return;
        int cx, cy;
        if (!WorldToCell(body.position, out cx, out cy)) return;
        int span = radius <= 0f ? 1 : Mathf.Max(1, Mathf.CeilToInt(radius / Mathf.Max(0.0001f, cellSize)));
        float rSqr = radius * radius;
        Vector2 pos = body.position;
        for (int oy = -span; oy <= span; oy++)
        {
            int ny = cy + oy;
            if (ny < 0 || ny >= _cellsY) continue;
            for (int ox = -span; ox <= span; ox++)
            {
                int nx = cx + ox;
                if (nx < 0 || nx >= _cellsX) continue;
                List<PhysicsBody> cell = _cells[ny * _cellsX + nx];
                for (int i = 0; i < cell.Count; i++)
                {
                    PhysicsBody other = cell[i];
                    if ((object)other == null) continue;
                    if (other.Equals(body)) continue;
                    if (radius > 0f && (other.position - pos).sqrMagnitude >= rSqr) continue;
                    outNeighbours.Add(other);
                }
            }
        }
    }

    public void GetNeighbours3x3(PhysicsBody body, List<PhysicsBody> outNeighbours)
    {
        outNeighbours.Clear();
        if ((object)body == null || _cells == null) return;
        int cx, cy;
        if (!WorldToCell(body.position, out cx, out cy)) return;
        for (int oy = -1; oy <= 1; oy++)
        {
            int ny = cy + oy;
            if (ny < 0 || ny >= _cellsY) continue;
            for (int ox = -1; ox <= 1; ox++)
            {
                int nx = cx + ox;
                if (nx < 0 || nx >= _cellsX) continue;
                List<PhysicsBody> cell = _cells[ny * _cellsX + nx];
                for (int i = 0; i < cell.Count; i++)
                {
                    PhysicsBody other = cell[i];
                    if ((object)other == null) continue;
                    if (other.Equals(body)) continue;
                    outNeighbours.Add(other);
                }
            }
        }
    }

    public List<PhysicsBody> GetCellContents(int cx, int cy)
    {
        if (_cells == null || !IsValidCell(cx, cy)) return null;
        return _cells[CellToIndex(cx, cy)];
    }

    public float GetDensity(PhysicsBody body)
    {
        int i;
        if (!_particleIndex.TryGetValue(body, out i)) return 0f;
        if (i < 0 || i >= _density.Count) return 0f;
        return _density[i];
    }

    public float GetPressure(PhysicsBody body)
    {
        int i;
        if (!_particleIndex.TryGetValue(body, out i)) return 0f;
        if (i < 0 || i >= _pressure.Count) return 0f;
        return _pressure[i];
    }

    public Vector2 GetSPHForce(PhysicsBody body)
    {
        int i;
        if (!_particleIndex.TryGetValue(body, out i)) return Vector2.zero;
        if (i < 0 || i >= _sphForce.Count) return Vector2.zero;
        return _sphForce[i];
    }

    void RebuildParticleIndex()
    {
        _particleIndex.Clear();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody b = _particles[i];
            if ((object)b == null) continue;
            _particleIndex[b] = i;
        }
    }

    void EnsureSphCapacity()
    {
        RebuildParticleIndex();
        int n = _particles.Count;
        while (_density.Count < n) _density.Add(0f);
        while (_pressure.Count < n) _pressure.Add(0f);
        while (_sphForce.Count < n) _sphForce.Add(Vector2.zero);
        if (_density.Count > n) _density.RemoveRange(n, _density.Count - n);
        if (_pressure.Count > n) _pressure.RemoveRange(n, _pressure.Count - n);
        if (_sphForce.Count > n) _sphForce.RemoveRange(n, _sphForce.Count - n);
    }

    Vector2 EstimateVelocity(PhysicsBody body, float dt)
    {
        Vector2 prev;
        if (!_prevPos.TryGetValue(body, out prev)) return Vector2.zero;
        if (dt <= 1e-6f) return Vector2.zero;
        return (body.position - prev) / dt;
    }

    public void ComputeSPHDensities()
    {
        EnsureSphCapacity();
        int n = _particles.Count;
        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h;
        float poly6Coef = 315f / (64f * Mathf.PI * Mathf.Pow(h, 9f));
        for (int i = 0; i < n; i++) _density[i] = 0f;
        for (int i = 0; i < n; i++)
        {
            PhysicsBody a = _particles[i];
            if ((object)a == null) continue;
            Vector2 pa = a.position;
            float rho = particleMass * poly6Coef * h2 * h2 * h2;
            int cx, cy;
            if (!WorldToCell(pa, out cx, out cy)) { _density[i] = rho; continue; }
            int span = NeighbourCellSpan();
            for (int oy = -span; oy <= span; oy++)
            {
                int ny = cy + oy;
                if (ny < 0 || ny >= _cellsY) continue;
                for (int ox = -span; ox <= span; ox++)
                {
                    int nx = cx + ox;
                    if (nx < 0 || nx >= _cellsX) continue;
                    List<PhysicsBody> cell = _cells[ny * _cellsX + nx];
                    for (int k = 0; k < cell.Count; k++)
                    {
                        PhysicsBody b = cell[k];
                        if ((object)b == null || b.Equals(a)) continue;
                        Vector2 d = pa - b.position;
                        float r2 = d.sqrMagnitude;
                        if (r2 >= h2 || r2 < 1e-12f) continue;
                        float diff = h2 - r2;
                        rho += particleMass * poly6Coef * diff * diff * diff;
                    }
                }
            }
            _density[i] = Mathf.Max(rho, 1e-5f);
        }
    }

    public void ComputeSPHPressures()
    {
        EnsureSphCapacity();
        for (int i = 0; i < _particles.Count; i++)
            _pressure[i] = gasConstant * (_density[i] - restDensity);
    }

    public Vector2 ComputeSPHForce(PhysicsBody body)
    {
        int i;
        if (!_particleIndex.TryGetValue(body, out i)) return Vector2.zero;
        if (_density.Count != _particles.Count) ComputeSPHDensities();
        if (_pressure.Count != _particles.Count) ComputeSPHPressures();
        return ComputeSPHForceByIndex(i);
    }

    Vector2 ComputeSPHForceByIndex(int i)
    {
        PhysicsBody a = _particles[i];
        if ((object)a == null || _cells == null) return Vector2.zero;
        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h;
        float h6 = h2 * h2 * h2;
        float spikyGradCoef = -45f / (Mathf.PI * h6);
        float viscLapCoef = 45f / (Mathf.PI * h6);
        Vector2 pa = a.position;
        Vector2 va = EstimateVelocity(a, _fixedDt);
        float rhoA = Mathf.Max(_density[i], 1e-5f);
        float pA = _pressure[i];
        Vector2 fP = Vector2.zero;
        Vector2 fV = Vector2.zero;
        int cx, cy;
        if (!WorldToCell(pa, out cx, out cy)) return Vector2.zero;
        int span = NeighbourCellSpan();
        for (int oy = -span; oy <= span; oy++)
        {
            int ny = cy + oy;
            if (ny < 0 || ny >= _cellsY) continue;
            for (int ox = -span; ox <= span; ox++)
            {
                int nx = cx + ox;
                if (nx < 0 || nx >= _cellsX) continue;
                List<PhysicsBody> cell = _cells[ny * _cellsX + nx];
                for (int k = 0; k < cell.Count; k++)
                {
                    PhysicsBody b = cell[k];
                    if ((object)b == null || b.Equals(a)) continue;
                    Vector2 delta = pa - b.position;
                    float r2 = delta.sqrMagnitude;
                    if (r2 >= h2 || r2 < 1e-12f) continue;
                    float r = Mathf.Sqrt(r2);
                    Vector2 dir = delta / r;
                    int j;
                    _particleIndex.TryGetValue(b, out j);
                    float rhoB = rhoA;
                    float pB = pA;
                    if (j >= 0 && j < _density.Count) rhoB = Mathf.Max(_density[j], 1e-5f);
                    if (j >= 0 && j < _pressure.Count) pB = _pressure[j];
                    if (enablePressure)
                    {
                        float gradMag = spikyGradCoef * (h - r) * (h - r);
                        float sharedP = (pA + pB) / (2f * rhoB);
                        fP += -particleMass * sharedP * gradMag * dir;
                    }
                    if (enableViscosity)
                    {
                        Vector2 vb = EstimateVelocity(b, _fixedDt);
                        float lap = viscLapCoef * (h - r);
                        fV += viscosity * particleMass * (vb - va) / rhoB * lap;
                    }
                }
            }
        }
        Vector2 f = fP + fV;
        if (maxForce > 0f && f.sqrMagnitude > maxForce * maxForce)
            f = f.normalized * maxForce;
        return f;
    }

    public void ComputeSPHForces()
    {
        EnsureSphCapacity();
        for (int i = 0; i < _particles.Count; i++)
            _sphForce[i] = ComputeSPHForceByIndex(i);
    }

    public void ApplySPH(PhysicsBody body)
    {
        Vector2 f = ComputeSPHForce(body);
        if (f.sqrMagnitude < 1e-10f) return;
        body.ApplyForce(f, body.position);
    }

    public void ApplySPHToAll()
    {
        if (_cells == null || _particles.Count == 0) return;
        _fixedDt = Application.isPlaying ? Mathf.Max(Time.fixedDeltaTime, 1e-5f) : 0.02f;
        ComputeSPHDensities();
        ComputeSPHPressures();
        ComputeSPHForces();
        for (int i = 0; i < _particles.Count; i++)
        {
            PhysicsBody body = _particles[i];
            if ((object)body == null) continue;
            Vector2 f = _sphForce[i];
            if (f.sqrMagnitude >= 1e-10f) body.ApplyForce(f, body.position);
            _prevPos[body] = body.position;
        }
    }

    #region Legacy repulsion API (obsolete wrappers around SPH)
    [System.Obsolete("Use smoothingRadius.")]
    public float repulsionRadius { get { return smoothingRadius; } set { smoothingRadius = value; } }
    [System.Obsolete("Use gasConstant.")]
    public float repulsionStrength { get { return gasConstant; } set { gasConstant = value; } }
    [System.Obsolete("Use enableSPH.")]
    public bool enableRepulsion { get { return enableSPH; } set { enableSPH = value; } }
    [System.Obsolete("Use ComputeSPHForce.")]
    public Vector2 ComputeRepulsionForce(PhysicsBody body) { return ComputeSPHForce(body); }
    [System.Obsolete("Use ApplySPH.")]
    public void ApplyRepulsion(PhysicsBody body) { ApplySPH(body); }
    [System.Obsolete("Use ApplySPHToAll.")]
    public void ApplyRepulsionToAll() { ApplySPHToAll(); }
    #endregion
    #endregion

    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.cyan;
        Vector3 center = new Vector3(gridOrigin.x + gridSize.x * 0.5f, gridOrigin.y + gridSize.y * 0.5f, 0f);
        Gizmos.DrawWireCube(center, new Vector3(gridSize.x, gridSize.y, 0.01f));
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
}

// BURST-NOTES (future optimization, no API change needed):
// 1. Positions/densities/pressures: gather into NativeArray<float2/float> once per step.
// 2. Grid: replace List<PhysicsBody>[] with counting-sort into NativeArray<int>
//    cellStart/cellCount + sorted NativeArray<int> particleIndex (classic grid build).
// 3. Density: IJobParallelFor over particles; each job reads only neighbour-cell
//    index ranges + the position array (read-only) and writes one float density.
// 4. Force: second IJobParallelFor reading positions + densities + pressures,
//    writing one float2 force. Span = ceil(h / cellSize), not fixed 3x3.
// 5. Apply: main-thread loop calls body.ApplyForce(force[i], pos[i]).
// Keeping ComputeSPHDensities/ComputeSPHPressures/ComputeSPHForces separate from
// ApplySPHToAll is what makes that swap mechanical.
