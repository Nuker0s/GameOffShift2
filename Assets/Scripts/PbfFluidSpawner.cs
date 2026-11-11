using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Code-only spawner (no prefab): creates Dynamic bodies via
/// PhysicsWorld.CreateBody(def) + body.CreateShape(CircleGeometry, shapeDef).
/// Zero Instantiate, zero GetComponent. Registers each body with PbfFluidLite.
///
/// SPACING IS THE ONE KNOB. Radius, mass, block layout (and, in PbfFluidLite, the smoothing radius and
/// rest density) all follow from it, so you can make the fluid sparser or denser by changing Spacing only:
///   particle count ~ 1 / spacing^2   and   cost per particle stays the same.
/// The ENGINE keeps owning gravity, damping, mass and collisions; PbfFluidLite only adds pressure + viscosity.
/// </summary>
[DefaultExecutionOrder(-5)]
public class PbfFluidSpawner : MonoBehaviour
{
    [Header("Refs")]
    [Tooltip("Sim to register with. Auto-found if null.")]
    public PbfFluidLite sim;

    [Header("Layout")]
    public Vector2 spawnOrigin = new Vector2(-4f, 0f);
    [Tooltip("Distance between particles at rest. Bigger = fewer, coarser particles = cheaper. 0.35 - 0.6 is a good range.")]
    public float spacing = 0.4f;
    [Tooltip("ON: block size is Area Size, and rows/cols are derived from Spacing (so the volume stays constant when you change Spacing). " +
             "OFF: use Rows x Cols as typed.")]
    public bool layoutByArea = true;
    public Vector2 areaSize = new Vector2(2.5f, 2.5f);
    public int rows = 6;
    public int cols = 6;
    [Tooltip("Random offset added to each spawn position. Breaks perfect lattice so SPH forces engage immediately.")]
    public float jitter = 0.08f;

    [Header("Body (code-only, no GameObjects)")]
    [Tooltip("ON: radius = Spacing x Radius Ratio. Keep the diameter at ~80-90% of spacing: if circles already touch at rest, " +
             "engine contacts do all the work and the fluid pressure never gets a chance to act.")]
    public bool radiusFromSpacing = true;
    [Range(0.2f, 0.49f)] public float radiusRatio = 0.42f;
    [Tooltip("Only used when Radius From Spacing is OFF.")]
    public float particleRadius = 0.17f;
    [Tooltip("ON: mass = Bulk Density x Spacing^2, so the fluid weighs the same however sparse you make it " +
             "(total mass = bulk density x area). OFF: use Particle Mass.")]
    public bool massFromBulkDensity = true;
    [Tooltip("kg per m^2 of fluid. 8 matches the old 0.5 kg particles at 0.25 spacing.")]
    public float bulkDensity = 8f;
    [Tooltip("Only used when Mass From Bulk Density is OFF. 0 = let the engine compute it from Density x area.")]
    public float particleMass = 1.28f;
    [Tooltip("Only used when the mass is left to the engine.")]
    public float particleDensity = 1f;
    public float particleFriction = 0f;
    public float particleBounciness = 0f;
    public Color particleColor = Color.cyan;
    public float particleLinearDamping = 0.05f;
    [Tooltip("The body's own gravity scale. The sim multiplies the fluid type's gravityMultiplier into it.")]
    public float particleGravityScale = 1f;
    public bool particleFreezeRotation = true;

    [Header("Type")]
    [Tooltip("Fluid type index into PbfFluidLite.types (gravityMultiplier/stiffness/viscosity).")]
    public int spawnTypeId = 0;

    [Header("Behaviour")]
    public bool spawnOnStart = true;
    public bool clearBeforeSpawn = true;
    [Tooltip("Tell the sim the spawn spacing/mass so its smoothing radius and rest density always match what was actually spawned.")]
    public bool calibrateSimOnSpawn = true;

    [Header("Debug")]
    public bool showLogs = true;
    public bool showSpawnGizmo = true;

    readonly List<PhysicsBody> spawned = new List<PhysicsBody>();
    public int SpawnedCount => spawned.Count;

    void Start()
    {
        if (spawnOnStart) Spawn();
    }

    // Bodies created in code have no scene owner: without this they would outlive the component
    // (e.g. across a scene reload) as ghosts that still cost physics time.
    void OnDestroy()
    {
        Despawn(false);
    }

    void ResolveLayout(out int nRows, out int nCols)
    {
        float sp = Mathf.Max(0.01f, spacing);
        nRows = layoutByArea ? Mathf.Max(1, Mathf.RoundToInt(areaSize.y / sp)) : Mathf.Max(1, rows);
        nCols = layoutByArea ? Mathf.Max(1, Mathf.RoundToInt(areaSize.x / sp)) : Mathf.Max(1, cols);
    }

    [ContextMenu("Spawn")]
    public void Spawn()
    {
        if (!EnsureSim()) return;
        if (clearBeforeSpawn) Despawn(false);

        float sp = Mathf.Max(0.01f, spacing);
        float r = Mathf.Max(0.02f, radiusFromSpacing ? sp * radiusRatio : particleRadius);
        float wantedMass = massFromBulkDensity ? bulkDensity * sp * sp : particleMass;
        ResolveLayout(out int nRows, out int nCols);

        // Tell the sim first: it derives its smoothing radius from the spacing.
        if (calibrateSimOnSpawn) sim.ConfigureReference(sp, wantedMass);

        if (showLogs && 2f * r > 0.9f * sp)
            Debug.LogWarning($"[Spawner] diameter {2f * r:F2} is > 90% of spacing {sp:F2}: circles touch at rest, so engine contacts " +
                             "will dominate and the fluid pressure has almost no room to act. Try radius <= " + (0.45f * sp).ToString("F2") + ".");

        int n = 0;
        float lastMass = wantedMass;
        for (int row = 0; row < nRows; row++)
            for (int col = 0; col < nCols; col++)
            {
                Vector2 p = spawnOrigin + new Vector2(col * sp, row * sp);
                if (jitter > 0f) p += new Vector2(Random.Range(-jitter, jitter), Random.Range(-jitter, jitter));
                PhysicsBody body = SpawnOne(p, r, wantedMass, typeId: spawnTypeId);
                if (!body.isValid) continue;
                lastMass = body.mass;
                n++;
            }

        if (showLogs)
            Debug.Log($"[Spawner] spawned {n} bodies ({nCols}x{nRows}) spacing {sp:F2}, radius {r:F3}, mass {lastMass:F3}, " +
                      $"smoothing radius {sim.EffectiveSmoothing:F2}, jitter {jitter}.");
    }

    bool EnsureSim()
    {
        if (sim == null) sim = FindFirstObjectByType<PbfFluidLite>();
        if (sim == null)
        {
            if (showLogs) Debug.LogWarning("[Spawner] PbfFluidLite not found.");
            return false;
        }
        return true;
    }

    public float EffectiveSpacing => Mathf.Max(0.01f, spacing);

    public float DefaultRadius()
    {
        return Mathf.Max(0.02f, radiusFromSpacing ? EffectiveSpacing * radiusRatio : particleRadius);
    }

    public float DefaultMass()
    {
        return massFromBulkDensity ? bulkDensity * EffectiveSpacing * EffectiveSpacing : particleMass;
    }

    [ContextMenu("SpawnOne")]
    /// <summary>
    /// Spawns a single particle. All properties are optional (null = inspector default, radius/mass follow Spacing when Radius From Spacing / Mass From Bulk Density are ON).
    /// Returns the spawned body (check body.isValid). Also usable from other scripts (e.g. Player spiral spawn: spawner.SpawnOne(pos)).
    /// </summary>
    public PhysicsBody SpawnOne(
        Vector2? position = null,
        float? radius = null,
        float? mass = null,
        float? density = null,
        float? friction = null,
        float? bounciness = null,
        Color? color = null,
        float? linearDamping = null,
        float? gravityScale = null,
        bool? freezeRotation = null,
        int? typeId = null)
    {
        if (!EnsureSim()) return default;
        var bodyDef = new PhysicsBodyDefinition
        {
            type = PhysicsBody.BodyType.Dynamic,
            position = position ?? spawnOrigin,
            rotation = PhysicsRotate.down,
            linearDamping = linearDamping ?? particleLinearDamping,
            gravityScale = gravityScale ?? particleGravityScale,
            constraints = (freezeRotation ?? particleFreezeRotation)
                ? PhysicsBody.BodyConstraints.Rotation
                : PhysicsBody.BodyConstraints.None,
        };
        var shapeDef = new PhysicsShapeDefinition
        {
            triggerEvents = true,
            density = Mathf.Max(0.01f, density ?? particleDensity),
            surfaceMaterial = new PhysicsShape.SurfaceMaterial
            {
                customColor = color ?? particleColor,
                friction = Mathf.Max(0f, friction ?? particleFriction),
                bounciness = Mathf.Max(0f, bounciness ?? particleBounciness),
            }
        };
        return SpawnOneInternal(
            PhysicsWorld.defaultWorld,
            bodyDef,
            shapeDef,
            Mathf.Max(0.02f, radius ?? DefaultRadius()),
            mass ?? DefaultMass(),
            typeId ?? spawnTypeId);
    }

    PhysicsBody SpawnOneInternal(
        PhysicsWorld world,
        PhysicsBodyDefinition bodyDef,
        PhysicsShapeDefinition shapeDef,
        float radius,
        float mass,
        int typeId)
    {
        PhysicsBody body = world.CreateBody(bodyDef);
        if (!body.isValid) return body;
        PhysicsShape shape = body.CreateShape(
            new CircleGeometry { center = Vector2.zero, radius = radius },
            shapeDef
        );
        if (!shape.isValid) { try { body.Destroy(0); } catch { } return default; }
        if (mass > 1e-6f) body.mass = mass;
        sim.AddParticle(body, typeId);
        spawned.Add(body);
        return body;
    }

    /// <summary>Spawn one particle from explicit engine defs. Optional radius/mass/typeId fall back to inspector defaults (spacing-aware).</summary>
    public PhysicsBody SpawnOne(
        PhysicsBodyDefinition bodyDef,
        PhysicsShapeDefinition shapeDef,
        float? radius = null,
        float? mass = null,
        int? typeId = null)
    {
        if (!EnsureSim()) return default;
        return SpawnOneInternal(
            PhysicsWorld.defaultWorld, bodyDef, shapeDef,
            Mathf.Max(0.02f, radius ?? DefaultRadius()),
            mass ?? DefaultMass(), typeId ?? spawnTypeId);
    }

    /// <summary>Spawn one particle from an explicit body def + default shape (optional shape props fall back to inspector).</summary>
    public PhysicsBody SpawnOne(
        PhysicsBodyDefinition bodyDef,
        float? radius = null,
        float? mass = null,
        float? density = null,
        float? friction = null,
        float? bounciness = null,
        Color? color = null,
        int? typeId = null)
    {
        if (!EnsureSim()) return default;
        var shapeDef = new PhysicsShapeDefinition
        {
            triggerEvents = true,
            density = Mathf.Max(0.01f, density ?? particleDensity),
            surfaceMaterial = new PhysicsShape.SurfaceMaterial
            {
                customColor = color ?? particleColor,
                friction = Mathf.Max(0f, friction ?? particleFriction),
                bounciness = Mathf.Max(0f, bounciness ?? particleBounciness),
            }
        };
        return SpawnOneInternal(
            PhysicsWorld.defaultWorld, bodyDef, shapeDef,
            Mathf.Max(0.02f, radius ?? DefaultRadius()),
            mass ?? DefaultMass(), typeId ?? spawnTypeId);
    }

    /// <summary>Spawn one particle from an explicit shape def + default body (optional body props fall back to inspector).</summary>
    public PhysicsBody SpawnOne(
        PhysicsShapeDefinition shapeDef,
        Vector2? position = null,
        float? radius = null,
        float? mass = null,
        float? linearDamping = null,
        float? gravityScale = null,
        bool? freezeRotation = null,
        int? typeId = null)
    {
        if (!EnsureSim()) return default;
        var bodyDef = new PhysicsBodyDefinition
        {
            type = PhysicsBody.BodyType.Dynamic,
            position = position ?? spawnOrigin,
            rotation = PhysicsRotate.down,
            linearDamping = linearDamping ?? particleLinearDamping,
            gravityScale = gravityScale ?? particleGravityScale,
            constraints = (freezeRotation ?? particleFreezeRotation)
                ? PhysicsBody.BodyConstraints.Rotation
                : PhysicsBody.BodyConstraints.None,
        };
        return SpawnOneInternal(
            PhysicsWorld.defaultWorld, bodyDef, shapeDef,
            Mathf.Max(0.02f, radius ?? DefaultRadius()),
            mass ?? DefaultMass(), typeId ?? spawnTypeId);
    }

    [ContextMenu("Despawn")]
    public void Despawn() => Despawn(true);
    public void Despawn(bool sync)
    {
        foreach (var b in spawned)
        {
            if (!b.isValid) continue;
            if (sim != null) sim.RemoveParticle(b);
            try { b.Destroy(0); } catch { }
        }
        spawned.Clear();
    }

    void OnDrawGizmos()
    {
        if (!showSpawnGizmo) return;
        Gizmos.color = new Color(1f, 0.6f, 0.2f, 0.6f);
        ResolveLayout(out int nRows, out int nCols);
        float sp = Mathf.Max(0.01f, spacing);
        float w = Mathf.Max(0.01f, nCols * sp);
        float hgt = Mathf.Max(0.01f, nRows * sp);
        Vector3 min = (Vector3)spawnOrigin;
        Gizmos.DrawWireCube(min + new Vector3(w / 2f, hgt / 2f, 0f), new Vector3(w, hgt, 0.1f));
    }
}
