using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>PBF LITE on Physics Core 2D. Reads body.position/linearVelocity/mass, predicts + projects, writes back. No parallel integrator.</summary>
[DefaultExecutionOrder(10)]
public class PbfFluidLite : MonoBehaviour
{
    [System.Serializable]
    public class FluidType
    {
        public string name = "Water";
        public float restDensity = 20f;
        [Range(0f, 1f)] public float stiffness = 0.8f;
        [Range(0f, 1f)] public float viscosity = 0.2f;
        public float gravityMultiplier = 1f;
    }

    [Header("Grid (predefined, NOT O(n^2))")]
    public Vector2 gridOrigin = new Vector2(-10f, -10f);
    public Vector2 gridSize = new Vector2(20f, 20f);
    public float cellSize = 0.5f;
    public float smoothingRadius = 0.5f;

    [Header("Solver")]
    [Range(1, 10)] public int iterations = 3;
    public bool enablePBF = true;
    public bool runInFixedUpdate = true;
    public float maxSpeed = 20f;
    public float maxCorrection = 0.3f;

    [Header("Types")]
    public List<FluidType> types = new List<FluidType>()
    {
        new FluidType() { name = "Water", restDensity = 20f, stiffness = 0.8f, viscosity = 0.2f, gravityMultiplier = 1f },
        new FluidType() { name = "Oil", restDensity = 15f, stiffness = 0.6f, viscosity = 0.6f, gravityMultiplier = 1f },
    };
    public int defaultType = 0;

    [Header("Source (drag PhysicsPose objects, never Finger sphere)")]
    public List<PhysicsPose> manualPoses = new List<PhysicsPose>();
    public List<PhysicsPose> autoSyncSource;
    public bool autoSyncEveryFixedUpdate = false;

    [Header("Calibration")]
    public float referenceSpacing = 0.25f;
    public float referenceMass = 0.5f;

    public void AddParticle(PhysicsBody body, int typeId)
    {
        if (!body.isValid) return;
        int t = ClampType(typeId);
        // Capture engine gravityScale as PBF multiplier, then take over gravity
        // so the engine doesn't apply it a second time on Dynamic bodies.
        try
        {
            float gs = body.gravityScale;
            if (float.IsNaN(gs) || float.IsInfinity(gs)) gs = 1f;
            bodyGravityScale[body] = gs;
            body.gravityScale = 0f;
        }
        catch { bodyGravityScale[body] = 1f; }
        if (bodyType.ContainsKey(body)) { bodyType[body] = t; if (!simBodies.Contains(body)) simBodies.Add(body); return; }
        bodyType.Add(body, t); simBodies.Add(body);
    }
    public void AddParticle(PhysicsBody body)
    {
        AddParticle(body, ClampType(defaultType));
    }
    public void AddPose(PhysicsPose pose, int typeId)
    {
        if (pose == null) return;
        if (!simPoses.Contains(pose)) simPoses.Add(pose);
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
        if (bodyType.ContainsKey(body)) bodyType[body] = t; else AddParticle(body, t);
    }
    public void RemoveParticle(PhysicsBody body)
    {
        try
        {
            if (body.isValid && bodyGravityScale.TryGetValue(body, out float gs))
                body.gravityScale = gs;
        }
        catch { }
        bodyGravityScale.Remove(body);
        bodyType.Remove(body); simBodies.Remove(body);
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
        foreach (var kv in bodyGravityScale)
        {
            try { PhysicsBody b = kv.Key; if (b.isValid) b.gravityScale = kv.Value; } catch { }
        }
        bodyGravityScale.Clear();
        bodyType.Clear(); simBodies.Clear(); simPoses.Clear(); manualPoses.Clear();
    }
    public void Clear() => ClearParticles();
    public int GetParticleType(PhysicsBody body)
    {
        if (!body.isValid) return ClampType(defaultType);
        return bodyType.TryGetValue(body, out int t) ? t : ClampType(defaultType);
    }
    [ContextMenu("Sync From List")]
    public void SyncFromList()
    {
        HashSet<PhysicsPose> wanted = new HashSet<PhysicsPose>();
        if (manualPoses != null) foreach (var p in manualPoses) if (p != null) wanted.Add(p);
        if (autoSyncSource != null) foreach (var p in autoSyncSource) if (p != null) wanted.Add(p);
        // Preserve previously added code-only bodies (spawner creates engine
        // bodies with no PhysicsPose owner); only merge pose bodies in.
        HashSet<PhysicsBody> keep = new HashSet<PhysicsBody>();
        foreach (var b in simBodies)
            if (b.isValid) keep.Add(b);
        simPoses.Clear();
        simBodies.Clear();
        foreach (var b in keep) simBodies.Add(b);
        int def = ClampType(defaultType);
        foreach (var p in wanted)
        {
            simPoses.Add(p);
            PhysicsBody b = p.body;
            if (!b.isValid) continue;
            if (!simBodies.Contains(b)) simBodies.Add(b);
            keep.Add(b);
            if (!bodyType.ContainsKey(b)) bodyType[b] = def;
            // Pose bodies are also PBF-driven: PBF applies gravity, so stop the
            // engine double-applying it. Cache original scale once.
            try
            {
                if (!bodyGravityScale.ContainsKey(b))
                    bodyGravityScale[b] = b.gravityScale;
                b.gravityScale = 0f;
            }
            catch { }
        }
        List<PhysicsBody> dead = null;
        foreach (var k in bodyType.Keys) if (!keep.Contains(k)) { if (dead == null) dead = new List<PhysicsBody>(); dead.Add(k); }
        if (dead != null) foreach (var d in dead) { bodyType.Remove(d); bodyGravityScale.Remove(d); }
        CleanPoses(manualPoses);
        EnsureScratch(simBodies.Count);
        if (showLogs) Debug.Log($"[PbfFluidLite] SyncFromList: {simBodies.Count} particles.");
    }
    void MigrateLegacyDefaults()
    {
        // Old code defaults (restDensity 40/30) were ~2x the calibrated value
        // for the spawner's spacing/mass, forcing C<=0 and disabling pressure.
        // Existing scenes keep serialized values, so patch them once here.
        if (types != null)
        {
            bool waterLegacy = types.Count > 0 && types[0] != null && Mathf.Approximately(types[0].restDensity, 40f);
            bool oilLegacy = types.Count > 1 && types[1] != null && Mathf.Approximately(types[1].restDensity, 30f);
            if (waterLegacy || oilLegacy)
            {
                float cal = ComputeCalibratedDensity();
                if (waterLegacy)
                {
                    types[0].restDensity = cal;
                    if (showLogs) Debug.Log($"[PbfFluidLite] Migrated Water restDensity 40 -> {cal:F2} (matches spacing/mass).");
                }
                if (oilLegacy)
                {
                    types[1].restDensity = cal * 0.75f;
                    if (showLogs) Debug.Log($"[PbfFluidLite] Migrated Oil restDensity 30 -> {cal * 0.75f:F2}.");
                }
            }
        }
    }
    void OnEnable() { MigrateLegacyDefaults(); SyncFromList(); ApplyDrawOptions(); }
    void OnValidate()
    {
        if (smoothingRadius < 0.05f) smoothingRadius = 0.05f;
        if (cellSize < 0.05f) cellSize = 0.05f;
        if (gridSize.x < 1f) gridSize.x = 1f;
        if (gridSize.y < 1f) gridSize.y = 1f;
        if (iterations < 1) iterations = 1; if (iterations > 10) iterations = 10;
        if (maxSpeed < 0.1f) maxSpeed = 0.1f;
        if (maxCorrection < 0.01f) maxCorrection = 0.01f;
        if (referenceSpacing < 0.01f) referenceSpacing = 0.01f;
        if (referenceMass <= 0f) referenceMass = 1f;
        if (types == null) types = new List<FluidType>();
        if (types.Count == 0) types.Add(new FluidType());
        defaultType = ClampType(defaultType);
        MigrateLegacyDefaults();
        ApplyDrawOptions();
    }
    void FixedUpdate()
    {
        if (!runInFixedUpdate || !enablePBF) return;
        float dt = Time.fixedDeltaTime; if (dt <= 0f) return;
        if (autoSyncEveryFixedUpdate) SyncFromList();
        else if (simBodies.Count == 0 && manualPoses.Count > 0) SyncFromList();
        else PruneInvalid();
        StepPBF(dt);
    }

    [Header("Debug")]
    public bool showParticlesViaPhysicsDraw = true;
    public bool showGridGizmo = true;
    public bool drawGridWhenUnselected = true;
    public bool showLogs = false;

    readonly Dictionary<PhysicsBody, int> bodyType = new Dictionary<PhysicsBody, int>();
    readonly Dictionary<PhysicsBody, float> bodyGravityScale = new Dictionary<PhysicsBody, float>();
    readonly List<PhysicsBody> simBodies = new List<PhysicsBody>();
    readonly List<PhysicsPose> simPoses = new List<PhysicsPose>();
    List<int>[] cells;
    int cellsX, cellsY;
    float effectiveCell = 0.5f;
    Vector2[] pred, velPred, startPos, deltaP, vNew;
    float[] rho, lambda, cachedMass, rho0Arr, stiffArr, viscArr;
    bool[] validCache;
    public int ParticleCount => simBodies.Count;
    public int ClampType(int t)
    {
        if (types == null || types.Count == 0) return 0;
        if (t < 0) return 0; if (t >= types.Count) return types.Count - 1; return t;
    }
    void CleanPoses(List<PhysicsPose> list)
    {
        if (list == null) return;
        for (int i = list.Count - 1; i >= 0; i--) if (list[i] == null) list.RemoveAt(i);
    }
    void PruneInvalid()
    {
        bool removed = false;
        for (int i = simBodies.Count - 1; i >= 0; i--)
        {
            if (simBodies[i].isValid) continue;
            PhysicsBody dead = simBodies[i];
            simBodies.RemoveAt(i);
            bodyType.Remove(dead);
            bodyGravityScale.Remove(dead);
            removed = true;
        }
        if (removed) EnsureScratch(simBodies.Count);
    }
    void StepPBF(float dt)
    {
        int n = simBodies.Count; if (n == 0) return;
        if (types == null || types.Count == 0) return;
        EnsureScratch(n);
        float h = Mathf.Max(0.05f, smoothingRadius);
        float h2 = h * h; float pi = Mathf.PI;
        float h9 = Mathf.Pow(h, 9);
        float poly6 = 315f / (64f * pi * h9);
        float h6 = h2 * h2 * h2;
        float spikyGrad = -45f / (pi * h6);
        float selfW = poly6 * h6;
        Vector2 gravity = GetGravity();
        for (int i = 0; i < n; i++)
        {
            PhysicsBody b0 = simBodies[i];
            bool valid = false;
            try { valid = b0.isValid; } catch { valid = false; }
            validCache[i] = valid;
            if (!valid) { rho[i] = 1f; rho0Arr[i] = 1f; lambda[i] = 0f; deltaP[i] = Vector2.zero; vNew[i] = Vector2.zero; continue; }
            int t = ClampType(GetParticleType(b0));
            FluidType ft = types[t];
            rho0Arr[i] = Mathf.Max(0.001f, ft.restDensity);
            stiffArr[i] = Mathf.Clamp01(ft.stiffness);
            viscArr[i] = Mathf.Clamp01(ft.viscosity);
            float gs = 1f;
            if (!bodyGravityScale.TryGetValue(b0, out gs) || float.IsNaN(gs) || float.IsInfinity(gs)) gs = 1f;
            float m = 1f; try { m = b0.mass; } catch { m = 1f; }
            if (m <= 0f || float.IsNaN(m) || float.IsInfinity(m)) m = 1f;
            cachedMass[i] = m;
            Vector2 x, v;
            try { x = b0.position; v = b0.linearVelocity; }
            catch { validCache[i] = false; continue; }
            startPos[i] = x;
            Vector2 vP = v + gravity * (ft.gravityMultiplier * gs) * dt;
            float sp = vP.magnitude;
            if (sp > maxSpeed && sp > 1e-6f) vP *= (maxSpeed / sp);
            velPred[i] = vP; pred[i] = x + vP * dt;
        }
        RebuildGrid(pred, n);
        float invDt = 1f / Mathf.Max(dt, 1e-6f);


        for (int it = 0; it < iterations; it++)
        {
            for (int i = 0; i < n; i++)
            {
                if (!validCache[i]) { rho[i] = rho0Arr[i]; lambda[i] = 0f; continue; }
                Vector2 pic = pred[i]; float rho0 = rho0Arr[i];
                float dens = cachedMass[i] * selfW;
                int ccx = Mathf.FloorToInt((pic.x - gridOrigin.x) / effectiveCell);
                int ccy = Mathf.FloorToInt((pic.y - gridOrigin.y) / effectiveCell);
                bool ins = !(ccx < 0 || ccy < 0 || ccx >= cellsX || ccy >= cellsY);
                if (ins) dens += DensityNeighbors(pic, i, h2, poly6);
                rho[i] = dens;
                float C = dens / rho0 - 1f; if (C < 0f) C = 0f;
                if (C <= 0f) { lambda[i] = 0f; continue; }
                Vector2 gI = Vector2.zero; float s2 = 0f;
                if (ins)
                {
                    float invRho0 = 1f / rho0;
                    for (int oy = -1; oy <= 1; oy++)
                    {
                        int ny = ccy + oy; if (ny < 0 || ny >= cellsY) continue;
                        for (int ox = -1; ox <= 1; ox++)
                        {
                            int nx = ccx + ox; if (nx < 0 || nx >= cellsX) continue;
                            var cell = cells[ny * cellsX + nx];
                            for (int k = 0; k < cell.Count; k++)
                            {
                                int j = cell[k]; if (j == i || !validCache[j]) continue;
                                Vector2 dd = pic - pred[j]; float r = dd.magnitude;
                                if (r >= h || r < 1e-6f) continue;
                                float hr = h - r;
                                Vector2 gW = spikyGrad * hr * hr * (dd / r);
                                Vector2 gij = gW * (cachedMass[j] * invRho0);
                                gI += gij; s2 += gij.sqrMagnitude;
                            }
                        }
                    }
                }
                s2 += gI.sqrMagnitude;
                lambda[i] = (s2 < 1e-6f) ? 0f : (-C / (s2 + 1e-6f));
            }
            for (int i = 0; i < n; i++)
            {
                if (!validCache[i]) { deltaP[i] = Vector2.zero; continue; }
                Vector2 pic = pred[i]; float li = lambda[i];
                Vector2 corr = Vector2.zero;
                int ccx = Mathf.FloorToInt((pic.x - gridOrigin.x) / effectiveCell);
                int ccy = Mathf.FloorToInt((pic.y - gridOrigin.y) / effectiveCell);
                bool ins = !(ccx < 0 || ccy < 0 || ccx >= cellsX || ccy >= cellsY);
                if (ins) corr = CorrectionNeighbors(pic, i, ccx, ccy, li, h, spikyGrad);
                // NOTE: lambda already contains 1/rho0. Old code did stiff/rho0
                // which damped pressure ~12-40x into invisibility. Stiffness is
                // now a plain 0..1 relaxation on the full PBF correction.
                corr *= stiffArr[i];
                float m = corr.magnitude;
                if (m > maxCorrection && m > 1e-9f) corr *= (maxCorrection / m);
                deltaP[i] = corr;
            }
            for (int i = 0; i < n; i++) if (validCache[i]) pred[i] += deltaP[i];
            if (it < iterations - 1) RebuildGrid(pred, n);
        }
        for (int i = 0; i < n; i++)
        {
            if (!validCache[i]) continue;
            vNew[i] = (pred[i] - startPos[i]) * invDt;
        }
        for (int i = 0; i < n; i++)
        {
            if (!validCache[i]) continue;
            float visc = viscArr[i];
            if (visc > 0f)
            {
                Vector2 pic = pred[i]; Vector2 acc = Vector2.zero; float wSum = 0f;
                Vector2 vi = vNew[i];
                int ccx = Mathf.FloorToInt((pic.x - gridOrigin.x) / effectiveCell);
                int ccy = Mathf.FloorToInt((pic.y - gridOrigin.y) / effectiveCell);
                bool ins = !(ccx < 0 || ccy < 0 || ccx >= cellsX || ccy >= cellsY);
                if (ins)
                {
                    for (int oy = -1; oy <= 1; oy++)
                    {
                        int ny = ccy + oy; if (ny < 0 || ny >= cellsY) continue;
                        for (int ox = -1; ox <= 1; ox++)
                        {
                            int nx = ccx + ox; if (nx < 0 || nx >= cellsX) continue;
                            var cell = cells[ny * cellsX + nx];
                            for (int k = 0; k < cell.Count; k++)
                            {
                                int j = cell[k]; if (j == i || !validCache[j]) continue;
                                Vector2 dd = pic - pred[j]; float r2 = dd.sqrMagnitude;
                                if (r2 >= h2 || r2 < 1e-9f) continue;
                                float q = h2 - r2; float w = poly6 * q * q * q;
                                acc += (vNew[j] - vi) * w; wSum += w;
                            }
                        }
                    }
                }
                if (wSum > 1e-9f) vNew[i] = vi + acc * (visc * 0.15f / (wSum + 1e-6f));
            }
            float sp2 = vNew[i].magnitude;
            if (sp2 > maxSpeed && sp2 > 1e-6f) vNew[i] *= (maxSpeed / sp2);
        }
        for (int i = 0; i < n; i++)
        {
            if (!validCache[i]) continue;
            PhysicsBody bw = simBodies[i];
            bool ok = false; try { ok = bw.isValid; } catch { ok = false; }
            if (!ok) { validCache[i] = false; continue; }
            try { bw.position = pred[i]; bw.linearVelocity = vNew[i]; } catch { }
        }
    }


    void RebuildGrid(Vector2[] pos, int n)
    {
        float cs = Mathf.Max(cellSize, smoothingRadius, 0.05f);
        effectiveCell = cs;
        cellsX = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cs));
        cellsY = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cs));
        int total = cellsX * cellsY;
        if (cells == null || cells.Length != total)
        { cells = new List<int>[total]; for (int k = 0; k < total; k++) cells[k] = new List<int>(); }
        else for (int k = 0; k < total; k++) cells[k].Clear();
        for (int i = 0; i < n; i++)
        {
            if (validCache != null && validCache.Length == n && !validCache[i]) continue;
            if (validCache == null && !simBodies[i].isValid) continue;
            int cx = Mathf.FloorToInt((pos[i].x - gridOrigin.x) / cs);
            int cy = Mathf.FloorToInt((pos[i].y - gridOrigin.y) / cs);
            if (cx < 0 || cy < 0 || cx >= cellsX || cy >= cellsY) continue;
            cells[cy * cellsX + cx].Add(i);
        }
    }
    float DensityNeighbors(Vector2 pic, int self, float h2, float poly6)
    {
        float dens = 0f;
        int ccx = Mathf.FloorToInt((pic.x - gridOrigin.x) / effectiveCell);
        int ccy = Mathf.FloorToInt((pic.y - gridOrigin.y) / effectiveCell);
        for (int oy = -1; oy <= 1; oy++)
        {
            int ny = ccy + oy; if (ny < 0 || ny >= cellsY) continue;
            for (int ox = -1; ox <= 1; ox++)
            {
                int nx = ccx + ox; if (nx < 0 || nx >= cellsX) continue;
                var cell = cells[ny * cellsX + nx];
                for (int k = 0; k < cell.Count; k++)
                {
                    int j = cell[k]; if (j == self || !validCache[j]) continue;
                    Vector2 d = pic - pred[j]; float r2 = d.sqrMagnitude;
                    if (r2 >= h2 || r2 < 1e-12f) continue;
                    float q = h2 - r2; dens += cachedMass[j] * poly6 * q * q * q;
                }
            }
        }
        return dens;
    }
    Vector2 CorrectionNeighbors(Vector2 pic, int self, int ccx, int ccy, float li, float h, float spikyGrad)
    {
        Vector2 corr = Vector2.zero;
        for (int oy = -1; oy <= 1; oy++)
        {
            int ny = ccy + oy; if (ny < 0 || ny >= cellsY) continue;
            for (int ox = -1; ox <= 1; ox++)
            {
                int nx = ccx + ox; if (nx < 0 || nx >= cellsX) continue;
                var cell = cells[ny * cellsX + nx];
                for (int k = 0; k < cell.Count; k++)
                {
                    int j = cell[k]; if (j == self || !validCache[j]) continue;
                    float lj = lambda[j];
                    if (li == 0f && lj == 0f) continue;
                    Vector2 d = pic - pred[j]; float r = d.magnitude;
                    if (r >= h || r < 1e-6f) continue;
                    float hr = h - r;
                    corr += spikyGrad * hr * hr * (d / r) * (li + lj);
                }
            }
        }
        return corr;
    }
    void EnsureScratch(int n)
    {
        if (pred != null && pred.Length == n) return;
        pred = new Vector2[n]; velPred = new Vector2[n]; startPos = new Vector2[n]; deltaP = new Vector2[n]; vNew = new Vector2[n];
        rho = new float[n]; lambda = new float[n]; cachedMass = new float[n];
        rho0Arr = new float[n]; stiffArr = new float[n]; viscArr = new float[n];
        validCache = new bool[n];
    }
    Vector2 GetGravity()
    {
        try { return PhysicsWorld.defaultWorld.gravity; }
        catch { }
        try { return Physics2D.gravity; } catch { }
        return new Vector2(0f, -9.81f);
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
    [ContextMenu("Calibrate Rest Density From Spacing")]
    public void CalibrateRestDensity()
    {
        float rho0 = ComputeCalibratedDensity();
        if (types == null || types.Count == 0) types = new List<FluidType>() { new FluidType() };
        foreach (var ft in types) ft.restDensity = rho0;
        Debug.Log($"[PbfFluidLite] restDensity={rho0:F3} (h={Mathf.Max(0.05f, smoothingRadius)} sp={Mathf.Max(0.01f, referenceSpacing)} mass={referenceMass}).");
    }
    float ComputeCalibratedDensity()
    {
        float h = Mathf.Max(0.05f, smoothingRadius); float h2 = h * h;
        float poly6 = 315f / (64f * Mathf.PI * Mathf.Pow(h, 9));
        float sp = Mathf.Max(0.01f, referenceSpacing); float sumW = 0f;
        for (float y = -h; y <= h + 1e-6f; y += sp)
            for (float x = -h; x <= h + 1e-6f; x += sp)
            { float r2 = x * x + y * y; if (r2 >= h2) continue; float q = h2 - r2; sumW += poly6 * q * q * q; }
        return Mathf.Max(0.001f, referenceMass * sumW);
    }
    void DrawGridLines()
    {
        float cs = Mathf.Max(cellSize, smoothingRadius, 0.05f);
        int nx = Mathf.Max(1, Mathf.CeilToInt(gridSize.x / cs));
        int ny = Mathf.Max(1, Mathf.CeilToInt(gridSize.y / cs));
        Vector2 min = gridOrigin; Vector2 max = gridOrigin + gridSize;
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.9f);
        Gizmos.DrawLine(new Vector3(min.x, min.y, 0f), new Vector3(max.x, min.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, min.y, 0f), new Vector3(max.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(max.x, max.y, 0f), new Vector3(min.x, max.y, 0f));
        Gizmos.DrawLine(new Vector3(min.x, max.y, 0f), new Vector3(min.x, min.y, 0f));
        Gizmos.color = new Color(0.2f, 0.8f, 1f, 0.35f);
        for (int i = 1; i < nx; i++)
        { float x = min.x + i * cs; if (x > max.x) break; Gizmos.DrawLine(new Vector3(x, min.y, 0f), new Vector3(x, max.y, 0f)); }
        for (int i = 1; i < ny; i++)
        { float y = min.y + i * cs; if (y > max.y) break; Gizmos.DrawLine(new Vector3(min.x, y, 0f), new Vector3(max.x, y, 0f)); }
    }
    void OnDrawGizmos() { if (!showGridGizmo || !drawGridWhenUnselected) return; DrawGridLines(); }
    void OnDrawGizmosSelected() { if (!showGridGizmo) return; DrawGridLines(); }
}
