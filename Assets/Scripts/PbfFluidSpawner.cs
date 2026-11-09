using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Code-only spawner (no prefab): creates Dynamic bodies via
/// PhysicsBody.Create(world, def) + body.CreateShape(CircleGeometry.Create(r), shapeDef).
/// Zero Instantiate, zero GetComponent. Registers each body with PbfFluidLite.
/// </summary>
[DefaultExecutionOrder(-5)]
public class PbfFluidSpawner : MonoBehaviour
{
    [Header("Refs")]
    [Tooltip("Sim to register with. Auto-found if null.")]
    public PbfFluidLite sim;
    [Header("Body (code-only, no GameObjects)")]
    public float particleRadius = 0.12f;
    public float particleMass = 0.5f;
    public float particleDensity = 1f;
    public float particleFriction = 0f;
    public float particleBounciness = 0f;
    public Color particleColor = Color.cyan;
    public float particleLinearDamping = 0.05f;
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

    [Header("Debug")]
    public bool showLogs = true;
    public bool showSpawnGizmo = true;

    readonly List<PhysicsBody> spawned = new List<PhysicsBody>();
    public int SpawnedCount => spawned.Count;

    void Start()
    {
        if (spawnOnStart) Spawn();
    }
    [ContextMenu("Spawn")]
    public void Spawn()
    {
        if (sim == null) sim = FindFirstObjectByType<PbfFluidLite>();
        if (sim == null) { Debug.LogWarning("[Spawner] PbfFluidLite not found."); return; }
        if (clearBeforeSpawn) Despawn(false);
        PhysicsWorld world = PhysicsWorld.defaultWorld;
        float r = Mathf.Max(0.02f, particleRadius);
        var shapeDef = new PhysicsShapeDefinition
        {
            triggerEvents = true, // Finger.cs GetTriggerVisitors() sees fluid
            density = Mathf.Max(0.01f, particleDensity),
            surfaceMaterial = new PhysicsShape.SurfaceMaterial
            {
                customColor = particleColor,
                friction = Mathf.Max(0f, particleFriction),
                bounciness = Mathf.Max(0f, particleBounciness),
            }
        };
        int n = 0;
        for (int row = 0; row < rows; row++)
            for (int col = 0; col < cols; col++)
            {
                Vector2 p = spawnOrigin + new Vector2(col * spacing, row * spacing);
                if (jitter > 0f) p += new Vector2(Random.Range(-jitter, jitter), Random.Range(-jitter, jitter));
                PhysicsBody body = world.CreateBody(
                    new PhysicsBodyDefinition
                    {
                        type = PhysicsBody.BodyType.Dynamic,
                        position = p,
                        rotation = PhysicsRotate.down,
                        linearDamping = particleLinearDamping,
                        gravityScale = particleGravityScale,
                        constraints = particleFreezeRotation
                            ? PhysicsBody.BodyConstraints.Rotation
                            : PhysicsBody.BodyConstraints.None,
                    }
                );
                if (!body.isValid) continue;
                PhysicsShape shape = body.CreateShape(
                    new CircleGeometry { center = Vector2.zero, radius = r },
                    shapeDef
                );
                if (!shape.isValid) { body.Destroy(0); continue; }
                if (particleMass > 1e-6f) body.mass = particleMass;
                sim.AddParticle(body, spawnTypeId); // PbfFluidLite: type selects gravityMultiplier/stiffness/viscosity
                spawned.Add(body);
                n++;
            }
        // Engine-created bodies have no scene PhysicsPose owner; SyncFromList()
        // now preserves code-only bodies, so no extra sync needed here.
        if (showLogs) Debug.Log($"[Spawner] spawned {n} code-only bodies into PbfFluidLite (spacing {spacing}, jitter {jitter}).");
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
        // Bodies removed directly; SyncFromList() now preserves code-only
        // bodies, so no extra sync is required here.
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
