using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using Unity.U2D.Physics;

public class Player : MonoBehaviour
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

    [Header("Solidification")]
    [Min(0.01f)]
    public float boundaryPadding = 0.25f;

    [Header("Liquid Movement")]
    public float liquidBaseForce = 3f;
    public Vector2 baseOffset = Vector2.zero;
    public float moveForce = 0f;
    public float maxForceDistance = 1.5f;

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

        state = State.Liquid;

        Vector2 center = useTransformAsCenter
            ? (Vector2)transform.position
            : spiralCenter;

        float step = spiralSpacing > 0f
            ? spiralSpacing
            : spawner.spacing;

        float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));

        particles.Clear();

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

            particles.Add(particle);
        }
    }

    public void Solidify()
    {
        if (state == State.Solid || particles.Count < 3)
            return;

        PhysicsBody result = ParticleSolidifier.Solidify(particles, sim, boundaryPadding);

        if (!result.isValid)
            return;

        solidifiedBody = result;
        solidifiedParticleCount = solidifiedBody.shapeCount;
        particles.Clear();
        state = State.Solid;
    }

    public void Liquify()
    {
        if (state != State.Solid || !solidifiedBody.isValid)
            return;

        List<PhysicsBody> restoredParticles = ParticleSolidifier.Liquify(solidifiedBody);

        if (restoredParticles.Count == 0)
            return;

        particles = restoredParticles;
        solidifiedBody = default;
        solidifiedParticleCount = 0;
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

        if (state == State.Liquid || !(particles.Count == 0)) LiquidMovement();

        
    }
    void LiquidMovement()
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
    }

    private void ApplySolidMovement()
    {
        if (!solidifiedBody.isValid)
            return;

        Vector2 center = solidifiedBody.worldCenterOfMass;
        Vector2 moveCenter = center;
        moveCenter += baseOffset;
        moveCenter += PIR.instance.move.ReadValue<Vector2>().normalized * moveForce;

        float distance = Vector2.Distance(center, moveCenter);
        Vector2 force =
            (moveCenter - center).normalized *
            liquidBaseForce *
            Mathf.Clamp(distance, 0f, maxForceDistance) *
            Mathf.Max(1, solidifiedParticleCount);

        solidifiedBody.ApplyForce(force, solidifiedBody.worldCenterOfMass);
    }

    private void OnDestroy()
    {
        if (solidifiedBody.isValid)
            ParticleSolidifier.DestroySolidifiedBody(solidifiedBody);
    }
    void OnDrawGizmos()
    {
        if (Application.isPlaying == true) if(solidifiedBody!= null) Gizmos.DrawCube(solidifiedBody.position,Vector3.one*0.3f);
    }
}
