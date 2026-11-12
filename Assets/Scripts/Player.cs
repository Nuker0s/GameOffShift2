
using System;
using System.Collections.Generic;
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

    private Dictionary<PhysicsBody, CircleGeometry> originalCircles =
        new Dictionary<PhysicsBody, CircleGeometry>();

    private List<PhysicsFixedJoint> solidJoints =
        new List<PhysicsFixedJoint>();

    [Header("Solidification")]
    [Range(8, 64)]
    public int boundarySegments = 32;

    [Min(0.01f)]
    public float boundaryPadding = 0.25f;

    [Tooltip("Fixed joint stiffness. Zero means maximum stiffness.")]
    public float jointFrequency = 0f;

    [Range(0f, 1f)]
    public float jointDamping = 1f;

    [Header("Liquid Movement")]
    public float liquidBaseForce = 3;
    public Vector2 baseOffset = Vector2.zero;
    public float moveForce = 0;
    public float maxForceDistance = 1.5f;

    private struct NeighborEdge
    {
        public int a;
        public int b;
        public Vector2 midpoint;

        public NeighborEdge(int a, int b, Vector2 midpoint)
        {
            this.a = a;
            this.b = b;
            this.midpoint = midpoint;
        }
    }

    void Start()
    {
        Application.targetFrameRate = 240;

        if (spawner == null)
            spawner = FindFirstObjectByType<PbfFluidSpawner>();

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
        originalCircles.Clear();

        for (int i = 0; i < initalParticleCount; i++)
        {
            float r = step * Mathf.Sqrt(i);
            float a = i * goldenAngle;

            Vector2 pos = center +
                new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * r;

            PhysicsBody particle = spawner.SpawnOne(pos);

            if (!particle.isValid)
                continue;

            Unity.Collections.NativeArray<PhysicsShape> shapes = particle.GetShapes();

            if (shapes.Length == 0)
                continue;

            particles.Add(particle);
            originalCircles.Add(particle, shapes[0].circleGeometry);
        }
    }

    // ---------------------------------------------------------
    // SOLIDIFY
    // ---------------------------------------------------------

    public void Solidify()
    {
        if (state == State.Solid)
            return;

        List<int> validIndices = new List<int>();
        List<Vector2> positions = new List<Vector2>();

        for (int i = 0; i < particles.Count; i++)
        {
            PhysicsBody particle = particles[i];

            if (!particle.isValid)
                continue;

            validIndices.Add(i);
            positions.Add(particle.position);
        }

        if (positions.Count < 3)
        {
            Debug.LogWarning("[Player] Need at least 3 valid particles.");
            return;
        }

        // Generate unoptimized Voronoi cells.
        List<Vector2>[] cells = GenerateVoronoiCells(positions);

        if (cells == null)
        {
            Debug.LogWarning("[Player] Failed to generate Voronoi cells.");
            return;
        }

        // Find neighboring cells before modifying any shapes.
        List<NeighborEdge> neighbors = FindNeighbors(cells, positions);

        // Validate all geometries before applying them.
        List<PolygonGeometry> geometries =
            new List<PolygonGeometry>(positions.Count);

        for (int i = 0; i < cells.Length; i++)
        {
            PhysicsBody particle = particles[validIndices[i]];

            List<Vector2> localVertices = new List<Vector2>();

            foreach (Vector2 worldVertex in cells[i])
            {
                localVertices.Add(particle.GetLocalPoint(worldVertex));
            }

            PolygonGeometry geometry =
                PolygonGeometry.Create(localVertices.ToArray());

            if (!geometry.isValid)
            {
                Debug.LogWarning(
                    "[Player] Invalid polygon generated for particle " + i);

                return;
            }

            geometries.Add(geometry);
        }

        // Replace circles with their respective polygon cells.
        for (int i = 0; i < geometries.Count; i++)
        {
            PhysicsBody particle = particles[validIndices[i]];
            Unity.Collections.NativeArray<PhysicsShape> shapes = particle.GetShapes();

            if (shapes.Length == 0)
                continue;

            PhysicsShape shape = shapes[0];
            shape.polygonGeometry = geometries[i];
        }

        // Connect every pair of neighboring cells.
        solidJoints.Clear();

        try
        {
            foreach (NeighborEdge edge in neighbors)
            {
                PhysicsBody a = particles[validIndices[edge.a]];
                PhysicsBody b = particles[validIndices[edge.b]];

                PhysicsFixedJointDefinition definition =
                    PhysicsFixedJointDefinition.defaultDefinition;

                definition.bodyA = a;
                definition.bodyB = b;

                // Both anchor frames meet at the shared cell boundary.
                definition.localAnchorA = new PhysicsTransform(
                    a.GetLocalPoint(edge.midpoint),
                    a.rotation.Inverse()
                );

                definition.localAnchorB = new PhysicsTransform(
                    b.GetLocalPoint(edge.midpoint),
                    b.rotation.Inverse()
                );

                definition.collideConnected = false;
                definition.tuningFrequency = jointFrequency;
                definition.tuningDamping = jointDamping;

                PhysicsFixedJoint joint =
                    PhysicsFixedJoint.Create(a.world, definition);

                if (joint.isValid)
                    solidJoints.Add(joint);
            }
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            // Roll back if joint creation fails.
            Liquify();
            return;
        }

        state = State.Solid;

        Debug.Log(
            "[Player] Solidified " + geometries.Count +
            " particles with " + solidJoints.Count + " fixed joints."
        );
    }

    // ---------------------------------------------------------
    // LIQUIFY
    // ---------------------------------------------------------

    public void Liquify()
    {
        // Destroy joints first.
        foreach (PhysicsFixedJoint joint in solidJoints)
        {
            if (joint.isValid)
                joint.Destroy();
        }

        solidJoints.Clear();

        // Restore original circle geometries.
        foreach (PhysicsBody particle in particles)
        {
            if (!particle.isValid)
                continue;

            if (!originalCircles.ContainsKey(particle))
                continue;

            Unity.Collections.NativeArray<PhysicsShape> shapes = particle.GetShapes();

            if (shapes.Length == 0)
                continue;

            PhysicsShape shape = shapes[0];
            shape.circleGeometry = originalCircles[particle];
        }

        state = State.Liquid;

        Debug.Log("[Player] Liquified.");
    }

    // ---------------------------------------------------------
    // VORONOI GENERATION
    // ---------------------------------------------------------

    private List<Vector2>[] GenerateVoronoiCells(List<Vector2> sites)
    {
        int count = sites.Count;

        Vector2 center = Vector2.zero;

        foreach (Vector2 point in sites)
            center += point;

        center /= count;

        float maxRadius = 0f;
        float averageNearestDistance = 0f;

        for (int i = 0; i < count; i++)
        {
            maxRadius = Mathf.Max(
                maxRadius,
                Vector2.Distance(center, sites[i])
            );

            float nearest = float.MaxValue;

            for (int j = 0; j < count; j++)
            {
                if (i == j)
                    continue;

                float distance = Vector2.Distance(sites[i], sites[j]);

                if (distance < 0.0001f)
                {
                    Debug.LogWarning(
                        "[Player] Duplicate particle positions detected."
                    );
                    return null;
                }

                nearest = Mathf.Min(nearest, distance);
            }

            averageNearestDistance += nearest;
        }

        averageNearestDistance /= count;

        float padding = Mathf.Max(
            boundaryPadding,
            averageNearestDistance * 0.8f
        );

        float radius = maxRadius + padding;

        // Shared convex boundary.
        List<Vector2> boundary = new List<Vector2>();

        for (int i = 0; i < boundarySegments; i++)
        {
            float angle = i * Mathf.PI * 2f / boundarySegments;

            boundary.Add(center + new Vector2(
                Mathf.Cos(angle),
                Mathf.Sin(angle)
            ) * radius);
        }

        List<Vector2>[] result = new List<Vector2>[count];

        // Clip the shared boundary against every other particle's
        // Voronoi half-plane.
        for (int i = 0; i < count; i++)
        {
            List<Vector2> polygon = new List<Vector2>(boundary);

            for (int j = 0; j < count; j++)
            {
                if (i == j)
                    continue;

                Vector2 normal = sites[j] - sites[i];

                float offset =
                    (sites[j].sqrMagnitude - sites[i].sqrMagnitude) * 0.5f;

                polygon = ClipPolygon(polygon, normal, offset);

                if (polygon.Count < 3)
                    return null;
            }

            // Physics Core 2D polygons have a maximum vertex count.
            SimplifyPolygon(polygon, PhysicsConstants.MaxPolygonVertices);

            if (polygon.Count < 3)
                return null;

            result[i] = polygon;
        }

        return result;
    }

    // Sutherland-Hodgman half-plane clipping.
    // Keeps points satisfying dot(point, normal) <= offset.
    private List<Vector2> ClipPolygon(
        List<Vector2> polygon,
        Vector2 normal,
        float offset)
    {
        List<Vector2> result = new List<Vector2>();

        if (polygon.Count == 0)
            return result;

        const float epsilon = 0.00001f;

        Vector2 previous = polygon[polygon.Count - 1];

        float previousDistance =
            Vector2.Dot(previous, normal) - offset;

        bool previousInside = previousDistance <= epsilon;

        foreach (Vector2 current in polygon)
        {
            float currentDistance =
                Vector2.Dot(current, normal) - offset;

            bool currentInside = currentDistance <= epsilon;

            if (currentInside != previousInside)
            {
                float t = previousDistance /
                    (previousDistance - currentDistance);

                Vector2 intersection = Vector2.Lerp(
                    previous,
                    current,
                    t
                );

                result.Add(intersection);
            }

            if (currentInside)
                result.Add(current);

            previous = current;
            previousDistance = currentDistance;
            previousInside = currentInside;
        }

        // Remove consecutive duplicate vertices.
        for (int i = result.Count - 1; i >= 0; i--)
        {
            int next = (i + 1) % result.Count;

            if (result.Count > 1 &&
                Vector2.Distance(result[i], result[next]) < 0.00001f)
            {
                result.RemoveAt(i);
            }
        }

        return result;
    }

    // Remove the least significant corners until the polygon
    // fits Physics Core 2D's vertex limit.
    private void SimplifyPolygon(List<Vector2> polygon, int maxVertices)
    {
        while (polygon.Count > maxVertices)
        {
            int removeIndex = -1;
            float smallestArea = float.MaxValue;

            for (int i = 0; i < polygon.Count; i++)
            {
                Vector2 previous =
                    polygon[(i - 1 + polygon.Count) % polygon.Count];

                Vector2 current = polygon[i];

                Vector2 next =
                    polygon[(i + 1) % polygon.Count];

                float area = Mathf.Abs(
                    (current.x - previous.x) * (next.y - previous.y) -
                    (current.y - previous.y) * (next.x - previous.x)
                );

                if (area < smallestArea)
                {
                    smallestArea = area;
                    removeIndex = i;
                }
            }

            if (removeIndex < 0)
                break;

            polygon.RemoveAt(removeIndex);
        }
    }

    // ---------------------------------------------------------
    // NEIGHBOR DETECTION
    // ---------------------------------------------------------

    private List<NeighborEdge> FindNeighbors(
        List<Vector2>[] cells,
        List<Vector2> sites)
    {
        List<NeighborEdge> result = new List<NeighborEdge>();

        float averageNearestDistance = 0f;

        for (int i = 0; i < sites.Count; i++)
        {
            float nearest = float.MaxValue;

            for (int j = 0; j < sites.Count; j++)
            {
                if (i == j)
                    continue;

                nearest = Mathf.Min(
                    nearest,
                    Vector2.Distance(sites[i], sites[j])
                );
            }

            averageNearestDistance += nearest;
        }

        averageNearestDistance /= sites.Count;

        float tolerance = Mathf.Max(
            0.0005f,
            averageNearestDistance * 0.002f
        );

        for (int i = 0; i < sites.Count; i++)
        {
            for (int j = i + 1; j < sites.Count; j++)
            {
                Vector2 direction = (sites[j] - sites[i]).normalized;

                Vector2 midpoint = (sites[i] + sites[j]) * 0.5f;

                List<Vector2> edgePoints = new List<Vector2>();

                // Find vertices on the shared Voronoi bisector.
                foreach (Vector2 vertex in cells[i])
                {
                    float distance = Mathf.Abs(
                        Vector2.Dot(vertex - midpoint, direction)
                    );

                    if (distance <= tolerance)
                        edgePoints.Add(vertex);
                }

                if (edgePoints.Count < 2)
                    continue;

                // Find the two furthest points on the shared edge.
                float longestDistance = 0f;
                Vector2 edgeMidpoint = Vector2.zero;

                for (int a = 0; a < edgePoints.Count; a++)
                {
                    for (int b = a + 1; b < edgePoints.Count; b++)
                    {
                        float distance = Vector2.Distance(
                            edgePoints[a],
                            edgePoints[b]
                        );

                        if (distance > longestDistance)
                        {
                            longestDistance = distance;
                            edgeMidpoint = (
                                edgePoints[a] + edgePoints[b]
                            ) * 0.5f;
                        }
                    }
                }

                if (longestDistance > tolerance * 2f)
                {
                    result.Add(new NeighborEdge(
                        i,
                        j,
                        edgeMidpoint
                    ));
                }
            }
        }

        return result;
    }

    void Update()
    {
        if(PIR.instance.stateChange.WasReleasedThisFrame())
        {
            Debug.Log("Test");
            if(state == State.Solid)
            {
                Debug.Log("Liquification");
                Liquify();
            }
            else if (state == State.Liquid)
            {
                Debug.Log("Solidification");
                Solidify();       
            }
            
        }
    }

    // ---------------------------------------------------------
    // LIQUID MOVEMENT
    // ---------------------------------------------------------

    void FixedUpdate()
    {
        if (state != State.Liquid || particles.Count == 0)
            return;

        Vector2 center = Vector2.zero;
        int validCount = 0;

        foreach (PhysicsBody part in particles)
        {
            if (!part.isValid)
                continue;

            center += part.position;
            validCount++;
        }

        if (validCount == 0)
            return;

        center /= validCount;

        Vector2 moveCenter = center;

        moveCenter += baseOffset;
        moveCenter += PIR.instance.move.ReadValue<Vector2>().normalized * moveForce;

        foreach (PhysicsBody part in particles)
        {
            if (!part.isValid)
                continue;

            float dist = Vector2.Distance(
                part.position,
                moveCenter
            );

            Vector2 force =
                (moveCenter - part.position).normalized *
                liquidBaseForce *
                Mathf.Clamp(dist, 0, maxForceDistance);

            part.ApplyForce(force, part.worldCenterOfMass);
        }
    }

    void OnDestroy()
    {
        foreach (PhysicsFixedJoint joint in solidJoints)
        {
            if (joint.isValid)
                joint.Destroy();
        }

        solidJoints.Clear();
    }
}
