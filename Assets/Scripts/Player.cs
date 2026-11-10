using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;
using System;
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
    List<PhysicsBody> particles = new List<PhysicsBody>();
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
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
    // Update is called once per frame
    void Update()
    {
        
    }
}
