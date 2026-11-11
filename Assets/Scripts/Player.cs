using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;
using System;
using Unity.VisualScripting;
/*public class Player : MonoBehaviour
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
    List<PhysicsBody> particles = new List<PhysicsBody>();

    public float liquidBaseForce = 3;
    public Vector2 baseOffset = Vector2.zero;
    public float moveForce=0;
    public float maxForceDistance = 1.5f;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
        if (spawner == null) spawner = FindFirstObjectByType<PbfFluidSpawner>();
        if (spawner == null)
        {
            Debug.LogWarning("[Player] No PbfFluidSpawner found, cannot spiral-spawn.");
            return;
        }

        Vector2 center = useTransformAsCenter ? (Vector2)transform.position : spiralCenter;
        float step = spiralSpacing > 0f ? spiralSpacing : spawner.spacing;
        // Golden-angle (phyllotaxis) spiral: even 2D packing, ideal for fluid start.
        float goldenAngle = Mathf.PI * (3f - Mathf.Sqrt(5f));

        particles.Clear();
        for (int i = 0; i < initalParticleCount; i++)
        {
            float r = step * Mathf.Sqrt(i);
            float a = i * goldenAngle;
            Vector2 pos = center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * r;
            PhysicsBody b = spawner.SpawnOne(pos);
            if (b.isValid) particles.Add(b);
        }
    }
    public void Solidify()
    {
        state = State.Solid;
        foreach (PhysicsBody particle in particles)
        {
            PhysicsShape shape = particle.GetShapes()[0];

            CircleGeometry geometry = shape.circleGeometry;
            geometry.radius = 0.5f;

            shape.circleGeometry = geometry;
        }
    }
    public void Liquify()
    {
        
    }
    // Update is called once per frame
    void FixedUpdate()
    {
        if (particles.Count == 0)
        {
            return;
        }

        // Calculate center of all the particles (average the position).
        Vector2 center = Vector2.zero;
        int validCount = 0;
        foreach (PhysicsBody part in particles)
        {
            if (!part.isValid)
            {
                continue;
            }

            Vector2 partPos = part.position; // that's how you get the position
            center += partPos;
            validCount++;
        }

        if (validCount == 0)
        {
            return;
        }

        center /= validCount;

        Vector2 moveCenter = center;

        moveCenter += baseOffset;
        moveCenter += PIR.instance.move.ReadValue<Vector2>().normalized * moveForce;
        foreach (PhysicsBody part in particles)
        {
            if (!part.isValid)
            {
                continue;
            }
            float dist = Vector2.Distance(part.position,moveCenter);
            // Force from the particle position to particle center.
            Vector2 force = (moveCenter - part.position).normalized * liquidBaseForce * Mathf.Clamp(dist,0,maxForceDistance);
            part.ApplyForce(force, part.worldCenterOfMass);
        }
    }
}
*/