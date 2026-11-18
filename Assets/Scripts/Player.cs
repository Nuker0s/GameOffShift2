using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using Unity.U2D.Physics;

public class Player : MonoBehaviour, PhysicsCallbacks.IContactCallback
{
    public PbfFluidLite sim;
    public PhysicsPose body;

    public int initalParticleCount = 50;

    public enum State
    {
        Solid,
        Liquid,
        Gas
    }

    public State state;
    public PbfFluidSpawner spawner;

    [Header("Spiral Spawn")]
    public Vector2 spiralCenter = Vector2.zero;
    public bool useTransformAsCenter = true;
    public float spiralSpacing = 0.25f;

    private List<PhysicsBody> particles = new List<PhysicsBody>();
    private PhysicsBody solidifiedBody;
    private int solidifiedParticleCount;

    // Keeps membership checks constant-time and prevents the same nearby
    // particle being absorbed by more than one player particle in one scan.
    private readonly HashSet<PhysicsBody> playerParticleSet = new HashSet<PhysicsBody>();
    private readonly List<PhysicsBody> nearbyParticleCandidates = new List<PhysicsBody>(64);
    private float absorptionTimer;

    [Header("Solidification")]
    [Min(0.01f)]
    public float boundaryPadding = 0.25f;

    [Header("Liquid Movement")]
    public float liquidBaseForce = 3f;
    public Vector2 baseOffset = Vector2.zero;
    public float moveForce = 0f;
    public float maxForceDistance = 1.5f;

    [Header("Solidified Shape Properties")]
    [Tooltip("Density applied to every generated polygon shape. Higher density increases the solidified body's mass.")]
    [Min(0.01f)]
    public float solidDensity = 5f;
    [Tooltip("Friction coefficient applied to every generated polygon shape. 0 = slippery, 1 = high friction.")]
    [Range(0f, 1f)]
    public float solidFriction = 0.9f;

    [Header("Solid Movement")]
    [Tooltip("Base torque multiplier. The actual torque is this value multiplied by the solidified body's final mass.")]
    [Min(0f)]
    public float solidTorque = 20f;

    [Header("Liquid Particle Absorption")]
    [Tooltip("Maximum center-to-center distance for absorbing a same-type particle.")]
    [Min(0.01f)]
    public float absorptionRange = 0.5f;
    [Tooltip("How often to query the simulation grid while liquid. 15 checks/sec is a good starting point.")]
    [Range(1f, 30f)]
    public float absorptionChecksPerSecond = 15f;

    private void Start()
    {
        Application.targetFrameRate = 240;

        if (spawner == null)
            spawner = FindFirstObjectByType<PbfFluidSpawner>();

        if (sim == null)
            sim = FindFirstObjectByType<PbfFluidLite>();

        if (spawner == null)
        {
            Debug.LogWarning("[Player] No PbfFluidSpawner found.");
            return;
        }

        if (sim == null)
            Debug.LogWarning("[Player] No PbfFluidLite found. Liquid particle absorption will be disabled.");

        state = State.Liquid;

        Vector2 center = useTransformAsCenter
            ? (Vector2)transform.position
            : spiralCenter;

        float step = spiralSpacing > 0f
            ? spiralSpacing
            : spawner.spacing;

        float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));

        particles.Clear();
        playerParticleSet.Clear();

        for (int i = 0; i < initalParticleCount; i++)
        {
            float radius = step * Mathf.Sqrt(i);
            float angle = i * goldenAngle;

            Vector2 position = center +
                new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius;

            PhysicsBody particle = spawner.SpawnOne(position);

            if (!particle.isValid)
                continue;

            bool hasOneCircle = false;

            using (NativeArray<PhysicsShape> shapes = particle.GetShapes())
            {
                hasOneCircle = shapes.Length == 1 &&
                    shapes[0].isValid &&
                    shapes[0].shapeType == PhysicsShape.ShapeType.Circle;
            }

            if (!hasOneCircle)
            {
                Debug.LogWarning("[Player] Spawned particle does not have exactly one Circle shape.");
                particle.Destroy();
                continue;
            }

            particle.SetContactEvents(true);

            using (NativeArray<PhysicsShape> shapes = particle.GetShapes())
            {
                for (int shapeIndex = 0; shapeIndex < shapes.Length; shapeIndex++)
                {
                    PhysicsShape shape = shapes[shapeIndex];
                    shape.callbackTarget = this;
                }
            }

            particles.Add(particle);
            playerParticleSet.Add(particle);
        }
    }

    public void Solidify()
    {
        if (state == State.Solid || particles.Count < 3)
            return;

        List<PhysicsBody> results = ParticleSolidifier.Solidify(
            particles, sim, 1f, boundaryPadding, null, "Terrain");

        if (results == null || results.Count == 0)
            return;

        PhysicsBody result = results[0];
        int numberofshapes = result.shapeCount;

        for (int i = 1; i < results.Count; i++)
        {
            if (results[i].shapeCount > numberofshapes)
            {
                result = results[i];
                numberofshapes = results[i].shapeCount;
            }
        }

        if (!result.isValid)
            return;

        solidifiedBody = result;
        ApplySolidifiedShapeProperties();
        solidifiedParticleCount = solidifiedBody.shapeCount;
        particles.Clear();
        playerParticleSet.Clear();
        absorptionTimer = 0f;
        state = State.Solid;
    }

    public void Liquify()
    {
        if (state != State.Solid || !solidifiedBody.isValid)
            return;

        List<PhysicsBody> restoredParticles =
            ParticleSolidifier.Liquify(solidifiedBody);

        if (restoredParticles.Count == 0)
            return;

        particles = restoredParticles;
        RebuildPlayerParticleSet();

        solidifiedBody = default;
        solidifiedParticleCount = 0;
        absorptionTimer = 0f;
        state = State.Liquid;
    }

    private void Update()
    {
        if (PIR.instance.stateChange.WasReleasedThisFrame())
        {
            if (state == State.Solid)
                Liquify();
            else if (state == State.Liquid)
                Solidify();
        }
    }

    private void FixedUpdate()
    {
        if (state == State.Solid)
        {
            ApplySolidMovement();
            return;
        }

        if (state == State.Liquid)
        {
            absorptionTimer += Time.fixedDeltaTime;
            float interval = 1f / Mathf.Max(1f, absorptionChecksPerSecond);

            if (absorptionTimer >= interval)
            {
                // Do one scan, not a burst of catch-up scans after a long frame.
                absorptionTimer %= interval;
                CheckForNearbyParticles();
            }
        }

        if (state == State.Liquid || particles.Count != 0)
            LiquidMovement();
    }

    private void LiquidMovement()
    {
        Vector2 center = Vector2.zero;
        int validCount = 0;

        foreach (PhysicsBody particle in particles)
        {
            if (!particle.isValid)
                continue;

            center += particle.position;
            validCount++;
        }

        if (validCount == 0)
            return;

        center /= validCount;

        Vector2 moveCenter = center;
        moveCenter += baseOffset;
        moveCenter += PIR.instance.move.ReadValue<Vector2>().normalized * moveForce;

        foreach (PhysicsBody particle in particles)
        {
            if (!particle.isValid)
                continue;

            float distance = Vector2.Distance(particle.position, moveCenter);

            Vector2 force =
                (moveCenter - particle.position).normalized *
                liquidBaseForce *
                Mathf.Clamp(distance, 0f, maxForceDistance);

            particle.ApplyForce(force, particle.worldCenterOfMass);
        }

        List<List<PhysicsBody>> results =
            ParticleBlobPartitioner.PartitionBlobs(particles.ToArray(), 1);

        if (results == null || results.Count == 0)
            return;

        List<PhysicsBody> result = results[0];
        int numberofshapes = result.Count;

        for (int i = 1; i < results.Count; i++)
        {
            if (results[i].Count > numberofshapes)
            {
                result = results[i];
                numberofshapes = result.Count;
            }
        }

        particles = result;
        RebuildPlayerParticleSet();
    }

    private void ApplySolidMovement()
    {
        if (!solidifiedBody.isValid)
            return;

        // PhysicsBody is 2D, so torque is a single scalar around Z.
        // Horizontal input spins the solidified body; contacts with the ground
        // or other geometry convert that rotation into rolling movement.
        float horizontalInput = PIR.instance.move.ReadValue<Vector2>().x;
        float bodyMass = Mathf.Max(0f, solidifiedBody.mass);
        float torque = -horizontalInput * solidTorque * bodyMass;

        if (Mathf.Abs(horizontalInput) > 0.001f && torque != 0f)
            solidifiedBody.ApplyTorque(torque);
    }

    private void ApplySolidifiedShapeProperties()
    {
        if (!solidifiedBody.isValid)
            return;

        float density = Mathf.Max(0.01f, solidDensity);
        float friction = Mathf.Clamp01(solidFriction);

        using (NativeArray<PhysicsShape> shapes = solidifiedBody.GetShapes())
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                PhysicsShape shape = shapes[i];
                if (!shape.isValid)
                    continue;

                // Defer the mass recalculation until every shape has its new density.
                shape.SetDensity(density, false);
                shape.friction = friction;
            }
        }

        // Recalculate once after updating all shapes so torque uses the final mass.
        solidifiedBody.ApplyMassFromShapes();
    }

    private void CheckForNearbyParticles()
    {
        if (sim == null || state != State.Liquid || particles.Count == 0)
            return;

        // Only particles that belonged to the player at the start of this scan
        // are query centers. Newly absorbed particles become centers next scan.
        int sourceCount = particles.Count;

        for (int i = 0; i < sourceCount; i++)
        {
            PhysicsBody playerParticle = particles[i];
            if (!playerParticle.isValid)
                continue;

            int playerType = sim.GetParticleType(playerParticle);
            sim.QueryParticlesInRange(
                playerParticle.position,
                absorptionRange,
                playerType,
                nearbyParticleCandidates);

            for (int candidateIndex = 0; candidateIndex < nearbyParticleCandidates.Count; candidateIndex++)
                TryAbsorbParticle(playerParticle, nearbyParticleCandidates[candidateIndex]);
        }
    }

    private void RebuildPlayerParticleSet()
    {
        playerParticleSet.Clear();

        for (int i = 0; i < particles.Count; i++)
        {
            PhysicsBody particle = particles[i];
            if (particle.isValid)
                playerParticleSet.Add(particle);
        }
    }

    // --------------------------------------------------
    // PARTICLE ABSORPTION
    // --------------------------------------------------

    private void TryAbsorbParticle(
        PhysicsBody playerParticle,
        PhysicsBody hitParticle)
    {
        if (state != State.Liquid || !playerParticle.isValid || !hitParticle.isValid)
            return;

        if (playerParticleSet.Contains(hitParticle))
            return;

        int playerType = sim.GetParticleType(playerParticle);
        int hitType = sim.GetParticleType(hitParticle);

        if (playerType < 0 || hitType < 0 || playerType != hitType)
            return;

        if (!playerParticleSet.Add(hitParticle))
            return;

        hitParticle.callbackTarget = this;
        particles.Add(hitParticle);

        Debug.Log("[Player] Absorbed particle. Total: " + particles.Count);
    }

    public void OnContactBegin2D(
        PhysicsEvents.ContactBeginEvent beginEvent)
    {
        // Keep contact absorption as an immediate supplement to the periodic grid query.
        if (sim == null || state != State.Liquid || particles.Count == 0)
            return;

        PhysicsBody bodyA = beginEvent.shapeA.body;
        PhysicsBody bodyB = beginEvent.shapeB.body;

        bool aBelongsToPlayer = playerParticleSet.Contains(bodyA);
        bool bBelongsToPlayer = playerParticleSet.Contains(bodyB);

        // Neither belongs to the player, or both already belong.
        if (aBelongsToPlayer == bBelongsToPlayer)
            return;

        if (aBelongsToPlayer)
            TryAbsorbParticle(bodyA, bodyB);
        else
            TryAbsorbParticle(bodyB, bodyA);
    }

    public void OnContactEnd2D(
        PhysicsEvents.ContactEndEvent endEvent)
    {
    }

    private void OnDestroy()
    {
        if (solidifiedBody.isValid)
            ParticleSolidifier.DestroySolidifiedBody(solidifiedBody);
    }

    private void OnDrawGizmos()
    {
        if (Application.isPlaying && solidifiedBody.isValid)
            Gizmos.DrawCube(solidifiedBody.position, Vector3.one * 0.3f);
    }
}
