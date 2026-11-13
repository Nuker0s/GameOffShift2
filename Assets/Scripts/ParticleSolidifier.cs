using System;
using System.Collections.Generic;
using Unity.Collections;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Runtime data attached to each polygon shape created by ParticleSolidifier.
/// PhysicsShapeDefinition cannot store arbitrary custom data, so the snapshot
/// is stored through PhysicsShape.userData.objectValue instead.
/// </summary>
public sealed class ParticleSolidifierSnapshot : ScriptableObject
{
    public CircleGeometry circleGeometry;
    public PhysicsShapeDefinition shapeDefinition;
    public PhysicsBodyDefinition bodyDefinition;

    // Particle pose relative to the solid body's origin/rotation.
    public Vector2 localPosition;
    public PhysicsRotate localRotation;

    // Preserve custom data and callback targets that belonged to the source.
    public PhysicsUserData shapeUserData;
    public PhysicsUserData bodyUserData;
    public UnityEngine.Object shapeCallbackTarget;
    public UnityEngine.Object bodyCallbackTarget;
}

/// <summary>
/// Converts a collection of individual circular particle bodies into one
/// compound PhysicsBody containing one Voronoi polygon shape per particle.
/// Liquify reconstructs the original particle bodies from data stored on
/// those polygon shapes.
/// </summary>
public static class ParticleSolidifier
{
    private sealed class ParticleInput
    {
        public PhysicsBody body;
        public PhysicsShape shape;
        public CircleGeometry circleGeometry;
        public PhysicsShapeDefinition shapeDefinition;
        public PhysicsBodyDefinition bodyDefinition;
        public Vector2 position;
        public PhysicsRotate rotation;
        public PhysicsUserData shapeUserData;
        public PhysicsUserData bodyUserData;
        public UnityEngine.Object shapeCallbackTarget;
        public UnityEngine.Object bodyCallbackTarget;
    }

    /// <summary>
    /// Solidifies particles into a single compound body.
    /// Each input body must have exactly one circular shape.
    ///
    /// boundaryPadding expands the convex hull before the Voronoi cells are
    /// generated. solidBodyDefinition optionally overrides the common body
    /// settings; its position, rotation and velocities are set by this method.
    /// </summary>
    public static PhysicsBody Solidify(
        List<PhysicsBody> particles,
        float boundaryPadding = 0.25f,
        PhysicsBodyDefinition? solidBodyDefinition = null)
    {
        if (particles == null || particles.Count < 3)
        {
            Debug.LogWarning("[ParticleSolidifier] At least 3 particles are required.");
            return default;
        }

        List<ParticleInput> inputs = new List<ParticleInput>(particles.Count);
        PhysicsWorld world = default;
        bool worldAssigned = false;

        for (int i = 0; i < particles.Count; i++)
        {
            PhysicsBody particle = particles[i];

            if (!particle.isValid)
                continue;

            if (particle.isOwned)
            {
                Debug.LogError(
                    "[ParticleSolidifier] Owned particle bodies cannot be destroyed without their owner key.");
                return default;
            }

            if (!worldAssigned)
            {
                world = particle.world;
                worldAssigned = true;
            }
            else if (!particle.world.Equals(world))
            {
                Debug.LogError("[ParticleSolidifier] All particles must belong to the same PhysicsWorld.");
                return default;
            }

            ParticleInput input = new ParticleInput();
            input.body = particle;
            input.position = particle.position;
            input.rotation = particle.rotation;
            input.bodyDefinition = particle.definition;
            input.bodyUserData = particle.userData;
            //input.bodyCallbackTarget = particle.callbackTarget;

            using (NativeArray<PhysicsShape> shapes = particle.GetShapes())
            {
                if (shapes.Length != 1)
                {
                    Debug.LogError(
                        "[ParticleSolidifier] Every particle body must have exactly one shape. Body " + i +
                        " has " + shapes.Length + ".");
                    return default;
                }

                PhysicsShape shape = shapes[0];

                if (!shape.isValid || shape.shapeType != PhysicsShape.ShapeType.Circle)
                {
                    Debug.LogError("[ParticleSolidifier] Every particle must have one valid Circle shape.");
                    return default;
                }

                input.shape = shape;
                input.circleGeometry = shape.circleGeometry;
                input.shapeDefinition = shape.definition;
                input.shapeUserData = shape.userData;
                //input.shapeCallbackTarget = shape.callbackTarget;
            }

            inputs.Add(input);
        }

        if (inputs.Count < 3)
        {
            Debug.LogWarning("[ParticleSolidifier] Fewer than 3 valid particle bodies were supplied.");
            return default;
        }

        List<Vector2> positions = new List<Vector2>(inputs.Count);
        Vector2 center = Vector2.zero;
        Vector2 averageVelocity = Vector2.zero;
        float averageAngularVelocity = 0f;

        for (int i = 0; i < inputs.Count; i++)
        {
            positions.Add(inputs[i].position);
            center += inputs[i].position;
            averageVelocity += inputs[i].body.linearVelocity;
            averageAngularVelocity += inputs[i].body.angularVelocity;
        }

        center /= inputs.Count;
        averageVelocity /= inputs.Count;
        averageAngularVelocity /= inputs.Count;

        List<Vector2>[] cells = GenerateVoronoiCells(positions, boundaryPadding);

        if (cells == null || cells.Length != inputs.Count)
        {
            Debug.LogWarning("[ParticleSolidifier] Failed to generate Voronoi cells.");
            return default;
        }

        // Validate every polygon before creating or destroying physics bodies.
        List<PolygonGeometry> geometries = new List<PolygonGeometry>(cells.Length);

        for (int i = 0; i < cells.Length; i++)
        {
            List<Vector2> localVertices = new List<Vector2>(cells[i].Count);

            for (int vertexIndex = 0; vertexIndex < cells[i].Count; vertexIndex++)
                localVertices.Add(cells[i][vertexIndex] - center);

            PolygonGeometry geometry = PolygonGeometry.Create(localVertices.ToArray());

            if (!geometry.isValid)
            {
                Debug.LogWarning("[ParticleSolidifier] Invalid Voronoi polygon generated for particle " + i + ".");
                return default;
            }

            geometries.Add(geometry);
        }

        PhysicsBodyDefinition bodyDefinition = solidBodyDefinition.HasValue
            ? solidBodyDefinition.Value
            : inputs[0].bodyDefinition;

        bodyDefinition.position = center;
        bodyDefinition.rotation = PhysicsRotate.identity;
        bodyDefinition.linearVelocity = averageVelocity;
        bodyDefinition.angularVelocity = averageAngularVelocity;

        PhysicsBody solidBody = world.CreateBody(bodyDefinition);

        if (!solidBody.isValid)
        {
            Debug.LogError("[ParticleSolidifier] Failed to create the solidified body.");
            return default;
        }

        List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>(inputs.Count);

        try
        {
            // Keep the body-level callback target where possible. Per-shape
            // callback targets are retained individually in their snapshots.
            solidBody.callbackTarget = inputs[0].bodyCallbackTarget;

            for (int i = 0; i < inputs.Count; i++)
            {
                ParticleInput input = inputs[i];
                ParticleSolidifierSnapshot snapshot =
                    ScriptableObject.CreateInstance<ParticleSolidifierSnapshot>();

                snapshot.hideFlags = HideFlags.HideAndDontSave;
                snapshot.circleGeometry = input.circleGeometry;
                snapshot.shapeDefinition = input.shapeDefinition;
                snapshot.bodyDefinition = input.bodyDefinition;
                snapshot.localPosition = input.position - center;
                snapshot.localRotation = solidBody.rotation.InverseMultiplyRotation(input.rotation);
                snapshot.shapeUserData = input.shapeUserData;
                snapshot.bodyUserData = input.bodyUserData;
                snapshot.shapeCallbackTarget = input.shapeCallbackTarget;
                snapshot.bodyCallbackTarget = input.bodyCallbackTarget;

                snapshots.Add(snapshot);

                PhysicsShapeDefinition polygonDefinition = input.shapeDefinition;
                polygonDefinition.startMassUpdate = false;

                PhysicsShape solidShape = solidBody.CreateShape(geometries[i], polygonDefinition);

                if (!solidShape.isValid)
                    throw new InvalidOperationException("Failed to create polygon shape " + i + ".");

                PhysicsUserData userData = solidShape.userData;
                userData.objectValue = snapshot;
                solidShape.userData = userData;
                solidShape.callbackTarget = input.shapeCallbackTarget;
            }

            // All shapes were added with mass updates disabled, so calculate
            // the compound body's mass properties once at the end.
            solidBody.ApplyMassFromShapes();

            // Only destroy source bodies after the replacement body is complete.
            for (int i = 0; i < inputs.Count; i++)
            {
                if (!inputs[i].body.Destroy())
                    throw new InvalidOperationException("Failed to destroy source particle body " + i + ".");
            }
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            if (solidBody.isValid)
                solidBody.Destroy();

            // Also destroys snapshots that may not have been attached yet
            // if shape creation failed partway through the loop.
            DestroySnapshots(snapshots);
            return default;
        }

        Debug.Log("[ParticleSolidifier] Solidified " + inputs.Count + " particles into one body.");
        return solidBody;
    }

    /// <summary>
    /// Recreates the original circular particle bodies from a solidified body.
    /// Particle positions follow the solid body's current pose. Their linear
    /// velocities are calculated from the solid body's velocity at each point.
    /// </summary>
    public static List<PhysicsBody> Liquify(PhysicsBody solidifiedBody)
    {
        List<PhysicsBody> particles = new List<PhysicsBody>();

        if (!solidifiedBody.isValid)
        {
            Debug.LogWarning("[ParticleSolidifier] Cannot liquify an invalid body.");
            return particles;
        }

        List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>();

        using (NativeArray<PhysicsShape> shapes = solidifiedBody.GetShapes())
        {
            if (shapes.Length == 0)
            {
                Debug.LogWarning("[ParticleSolidifier] The solidified body has no shapes.");
                return particles;
            }

            for (int i = 0; i < shapes.Length; i++)
            {
                PhysicsUserData userData = shapes[i].userData;
                ParticleSolidifierSnapshot snapshot = userData.objectValue as ParticleSolidifierSnapshot;

                if (snapshot == null)
                {
                    Debug.LogError(
                        "[ParticleSolidifier] Shape " + i + " has no particle snapshot. " +
                        "The body was not destroyed.");
                    return new List<PhysicsBody>();
                }

                snapshots.Add(snapshot);
            }
        }

        try
        {
            for (int i = 0; i < snapshots.Count; i++)
            {
                ParticleSolidifierSnapshot snapshot = snapshots[i];

                Vector2 worldPosition = solidifiedBody.GetWorldPoint(snapshot.localPosition);
                PhysicsRotate worldRotation =
                    solidifiedBody.rotation.MultiplyRotation(snapshot.localRotation);

                PhysicsBodyDefinition bodyDefinition = snapshot.bodyDefinition;
                bodyDefinition.position = worldPosition;
                bodyDefinition.rotation = worldRotation;
                bodyDefinition.linearVelocity = solidifiedBody.GetWorldPointVelocity(worldPosition);
                bodyDefinition.angularVelocity = solidifiedBody.angularVelocity;

                PhysicsBody particle = solidifiedBody.world.CreateBody(bodyDefinition);

                if (!particle.isValid)
                    throw new InvalidOperationException("Failed to recreate particle body " + i + ".");

                particle.userData = snapshot.bodyUserData;
                particle.callbackTarget = snapshot.bodyCallbackTarget;

                PhysicsShape particleShape = particle.CreateShape(
                    snapshot.circleGeometry,
                    snapshot.shapeDefinition);

                if (!particleShape.isValid)
                {
                    particle.Destroy();
                    throw new InvalidOperationException("Failed to recreate particle shape " + i + ".");
                }

                particleShape.userData = snapshot.shapeUserData;
                particleShape.callbackTarget = snapshot.shapeCallbackTarget;
                particles.Add(particle);
            }
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            for (int i = 0; i < particles.Count; i++)
            {
                if (particles[i].isValid)
                    particles[i].Destroy();
            }

            // Keep the solid body and its snapshots intact so the operation
            // can be retried or inspected after a failed reconstruction.
            return new List<PhysicsBody>();
        }

        // Particle reconstruction succeeded; now the old compound body and
        // its temporary snapshot objects can be removed.
        solidifiedBody.Destroy();
        DestroySnapshots(snapshots);

        Debug.Log("[ParticleSolidifier] Liquified " + particles.Count + " particles.");
        return particles;
    }

    /// <summary>
    /// Destroys a solidified body and the temporary snapshot objects stored
    /// on its shapes. Use this instead of PhysicsBody.Destroy when discarding
    /// a solidified body without liquifying it.
    /// </summary>
    public static void DestroySolidifiedBody(PhysicsBody solidifiedBody)
    {
        if (!solidifiedBody.isValid)
            return;

        List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>();

        using (NativeArray<PhysicsShape> shapes = solidifiedBody.GetShapes())
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                PhysicsUserData userData = shapes[i].userData;
                ParticleSolidifierSnapshot snapshot = userData.objectValue as ParticleSolidifierSnapshot;

                if (snapshot != null)
                    snapshots.Add(snapshot);
            }
        }

        solidifiedBody.Destroy();
        DestroySnapshots(snapshots);
    }

    private static void DestroySnapshots(List<ParticleSolidifierSnapshot> snapshots)
    {
        for (int i = 0; i < snapshots.Count; i++)
        {
            if (snapshots[i] != null)
                UnityEngine.Object.Destroy(snapshots[i]);
        }
    }

    // ---------------------------------------------------------------------
    // VORONOI GENERATION
    // ---------------------------------------------------------------------

    private static List<Vector2>[] GenerateVoronoiCells(List<Vector2> sites, float boundaryPadding)
    {
        int count = sites.Count;

        if (count < 3)
            return null;

        float averageNearestDistance = 0f;

        for (int i = 0; i < count; i++)
        {
            float nearest = float.MaxValue;

            for (int j = 0; j < count; j++)
            {
                if (i == j)
                    continue;

                float distance = Vector2.Distance(sites[i], sites[j]);

                if (distance < 0.0001f)
                {
                    Debug.LogWarning("[ParticleSolidifier] Duplicate particle positions detected.");
                    return null;
                }

                nearest = Mathf.Min(nearest, distance);
            }

            averageNearestDistance += nearest;
        }

        averageNearestDistance /= count;

        float padding = Mathf.Max(
            Mathf.Max(0.01f, boundaryPadding),
            averageNearestDistance * 0.8f);

        List<Vector2> boundary = GenerateConvexHull(sites);

        if (boundary.Count < 3)
            return null;

        boundary = ExpandConvexPolygon(boundary, padding);

        List<Vector2>[] result = new List<Vector2>[count];

        for (int i = 0; i < count; i++)
        {
            List<Vector2> polygon = new List<Vector2>(boundary);

            for (int j = 0; j < count; j++)
            {
                if (i == j)
                    continue;

                Vector2 normal = sites[j] - sites[i];
                float offset = (sites[j].sqrMagnitude - sites[i].sqrMagnitude) * 0.5f;

                polygon = ClipPolygon(polygon, normal, offset);

                if (polygon.Count < 3)
                    return null;
            }

            SimplifyPolygon(polygon, PhysicsConstants.MaxPolygonVertices);

            if (polygon.Count < 3)
                return null;

            result[i] = polygon;
        }

        return result;
    }

    private static List<Vector2> ExpandConvexPolygon(List<Vector2> polygon, float padding)
    {
        List<Vector2> expanded = new List<Vector2>();
        int count = polygon.Count;

        for (int i = 0; i < count; i++)
        {
            Vector2 previous = polygon[(i - 1 + count) % count];
            Vector2 current = polygon[i];
            Vector2 next = polygon[(i + 1) % count];

            Vector2 previousEdge = (current - previous).normalized;
            Vector2 nextEdge = (next - current).normalized;

            Vector2 previousNormal = new Vector2(previousEdge.y, -previousEdge.x);
            Vector2 nextNormal = new Vector2(nextEdge.y, -nextEdge.x);

            Vector2 previousPoint = current + previousNormal * padding;
            Vector2 nextPoint = current + nextNormal * padding;

            float denominator = Cross(previousEdge, nextEdge);

            if (Mathf.Abs(denominator) < 0.00001f)
            {
                Vector2 normal = (previousNormal + nextNormal).normalized;
                expanded.Add(current + normal * padding);
                continue;
            }

            float t = Cross(nextPoint - previousPoint, nextEdge) / denominator;
            expanded.Add(previousPoint + previousEdge * t);
        }

        return expanded;
    }

    private static List<Vector2> GenerateConvexHull(List<Vector2> points)
    {
        List<Vector2> sorted = new List<Vector2>(points);

        sorted.Sort((a, b) =>
        {
            int comparison = a.x.CompareTo(b.x);

            if (comparison == 0)
                comparison = a.y.CompareTo(b.y);

            return comparison;
        });

        List<Vector2> hull = new List<Vector2>();

        // Lower hull.
        for (int i = 0; i < sorted.Count; i++)
        {
            Vector2 point = sorted[i];

            while (hull.Count >= 2 &&
                Cross(
                    hull[hull.Count - 1] - hull[hull.Count - 2],
                    point - hull[hull.Count - 1]) <= 0f)
            {
                hull.RemoveAt(hull.Count - 1);
            }

            hull.Add(point);
        }

        // Upper hull.
        int lowerCount = hull.Count;

        for (int i = sorted.Count - 2; i >= 0; i--)
        {
            Vector2 point = sorted[i];

            while (hull.Count > lowerCount &&
                Cross(
                    hull[hull.Count - 1] - hull[hull.Count - 2],
                    point - hull[hull.Count - 1]) <= 0f)
            {
                hull.RemoveAt(hull.Count - 1);
            }

            hull.Add(point);
        }

        if (hull.Count > 1)
            hull.RemoveAt(hull.Count - 1);

        return hull;
    }

    private static float Cross(Vector2 a, Vector2 b)
    {
        return a.x * b.y - a.y * b.x;
    }

    // Sutherland-Hodgman clipping. Keeps points satisfying dot(point, normal) <= offset.
    private static List<Vector2> ClipPolygon(List<Vector2> polygon, Vector2 normal, float offset)
    {
        List<Vector2> result = new List<Vector2>();

        if (polygon.Count == 0)
            return result;

        const float epsilon = 0.00001f;

        Vector2 previous = polygon[polygon.Count - 1];
        float previousDistance = Vector2.Dot(previous, normal) - offset;
        bool previousInside = previousDistance <= epsilon;

        for (int i = 0; i < polygon.Count; i++)
        {
            Vector2 current = polygon[i];
            float currentDistance = Vector2.Dot(current, normal) - offset;
            bool currentInside = currentDistance <= epsilon;

            if (currentInside != previousInside)
            {
                float t = previousDistance / (previousDistance - currentDistance);
                result.Add(Vector2.Lerp(previous, current, t));
            }

            if (currentInside)
                result.Add(current);

            previous = current;
            previousDistance = currentDistance;
            previousInside = currentInside;
        }

        // Remove consecutive duplicate vertices.
        for (int i = result.Count - 1; i >= 0 && result.Count > 1; i--)
        {
            int next = (i + 1) % result.Count;

            if (Vector2.Distance(result[i], result[next]) < 0.00001f)
                result.RemoveAt(i);
        }

        return result;
    }

    // Remove the least significant corners until the polygon fits the vertex limit.
    private static void SimplifyPolygon(List<Vector2> polygon, int maxVertices)
    {
        while (polygon.Count > maxVertices)
        {
            int removeIndex = -1;
            float smallestArea = float.MaxValue;

            for (int i = 0; i < polygon.Count; i++)
            {
                Vector2 previous = polygon[(i - 1 + polygon.Count) % polygon.Count];
                Vector2 current = polygon[i];
                Vector2 next = polygon[(i + 1) % polygon.Count];

                float area = Mathf.Abs(
                    (current.x - previous.x) * (next.y - previous.y) -
                    (current.y - previous.y) * (next.x - previous.x));

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
}
