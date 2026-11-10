using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Code-only spawner (no prefab): creates Dynamic bodies via
/// PhysicsWorld.CreateBody(def) + body.CreateShape(CircleGeometry, shapeDef).
/// Zero Instantiate, zero GetComponent. Registers each body with PbfFluidLite.
/// The ENGINE keeps owning gravity, damping, mass and collisions for these bodies;
/// PbfFluidLite only adds pressure + viscosity as forces.
/// </summary>
[DefaultExecutionOrder(-5)]
public class PbfFluidSpawner : MonoBehaviour
{
    [Header("Refs")]
    [Tooltip("Sim to register with. Auto-found if null.")]
    public PbfFluidLite sim;
    [Header("Body (code-only, no GameObjects)")]
    [Tooltip("Keep diameter (2r) at about 80-90% of Spacing. If the circles already touch at rest, the engine's " +
             "contacts do all the work and the fluid pressure never gets a chance to act.")]
    public float particleRadius = 0.1f;
    [Tooltip("If > 0 the body mass is forced to this value. If 0 the engine computes it from Density x area.")]
    public float particleMass = 0.5f;
    [Tooltip("Only used when Particle Mass is 0.")]
    public float particleDensity = 1f;
    public float particleFriction = 0f;
    public float particleBounciness = 0f;
    public Color particleColor = Color.cyan;
    public float particleLinearDamping = 0.05f;
    [Tooltip("The body's own gravity scale. The sim multiplies the fluid type's gravityMultiplier into it.")]
    public float particleGravityScale = 1f;
    public bool particleFreezeRotation = true;

    [Header("Layout")]
    public Vector2 spawnOrigin = new Vector2(-4f, 0f);
    public int rows = 10;
    public int cols = 10;
    public float spacing = 0.25f;
    [Tooltip("Random offset added to each spawn position. Breaks perfect lattice so SPH forces engage immediately.")]
    public float jitter = 0.06f;

    [Header("Type")]
    [Tooltip("Fluid type index into PbfFluidLite.types (gravityMultiplier/stiffness/viscosity).")]
    public int spawnTypeId = 0;

    [Header("Behaviour")]
    public bool spawnOnStart = true;
    public bool clearBeforeSpawn = true;
    [Tooltip("Tell the sim the spawn spacing/mass so its rest density always matches what was actually spawned.")]
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
    // (e.g. across a scene reload) as invisible-to-the-sim ghosts that still cost physics time.
    void OnDestroy()
    {
        Despawn(false);
    }

    [ContextMenu("SpawnOne")]
    /// <summary>
    /// Spawns a single particle. All properties are optional (null = inspector default).
    /// Returns the spawned body (check body.isValid). Also usable from other scripts.
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
            triggerEvents = true, // Finger.cs GetTriggerVisitors() sees fluid
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
            Mathf.Max(0.02f, radius ?? particleRadius),
            mass ?? particleMass,
            typeId ?? spawnTypeId);
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

    /// <summary>Spawn one particle from explicit engine defs. Optional radius/mass/typeId fall back to inspector defaults.</summary>
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
            Mathf.Max(0.02f, radius ?? particleRadius),
            mass ?? particleMass, typeId ?? spawnTypeId);
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
            Mathf.Max(0.02f, radius ?? particleRadius),
            mass ?? particleMass, typeId ?? spawnTypeId);
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
            Mathf.Max(0.02f, radius ?? particleRadius),
            mass ?? particleMass, typeId ?? spawnTypeId);
    }

    [ContextMenu("Spawn")]
    public void Spawn()
    {
        if (!EnsureSim()) return;
        if (clearBeforeSpawn) Despawn(false);

        float r = Mathf.Max(0.02f, particleRadius);
        if (showLogs && 2f * r > 0.9f * spacing)
            Debug.LogWarning($"[Spawner] diameter {2f * r:F2} is > 90% of spacing {spacing:F2}: circles touch at rest, so engine contacts " +
                             "will dominate and the fluid pressure has almost no room to act. Try radius <= " + (0.45f * spacing).ToString("F2") + ".");

        int n = 0;
        float lastMass = particleMass;
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
            {
                Vector2 p = spawnOrigin + new Vector2(col * spacing, row * spacing);
                if (jitter > 0f) p += new Vector2(Random.Range(-jitter, jitter), Random.Range(-jitter, jitter));
                PhysicsBody body = SpawnOne(p);
                if (!body.isValid) continue;
                lastMass = body.mass;
                n++;
            }

        if (n > 0 && calibrateSimOnSpawn) sim.ConfigureReference(spacing, lastMass);
        if (showLogs) Debug.Log($"[Spawner] spawned {n} code-only bodies into PbfFluidLite (spacing {spacing}, jitter {jitter}, body mass {lastMass:F3}).");
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
        float w = Mathf.Max(0.01f, cols * spacing);
        float hgt = Mathf.Max(0.01f, rows * spacing);
        Vector3 min = (Vector3)spawnOrigin;
        Gizmos.DrawWireCube(min + new Vector3(w / 2f, hgt / 2f, 0f), new Vector3(w, hgt, 0.1f));
    }
}
