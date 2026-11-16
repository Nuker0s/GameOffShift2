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
    public object shapeCallbackTarget;
    public object bodyCallbackTarget;

    // Fluid-solver registration to restore when the particle is recreated.
    public PbfFluidLite simulation;
    public int fluidTypeId;

    // Mass/density overrides.
    public float mass;
    public float density;
}

/// <summary>
/// Body-level snapshot manifest. It is required because a particle whose cell
/// is completely removed by terrain has no output shape on which to store its
/// snapshot. Every original particle snapshot is stored here, including those
/// which produce zero resulting physics shapes.
/// </summary>
public sealed class ParticleSolidifierManifest : ScriptableObject
{
    public List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>();
}

/// <summary>
/// Converts a collection of individual circular particle bodies into one
/// compound PhysicsBody. Each particle's Voronoi cell can become multiple
/// convex polygon shapes after terrain subtraction. All shapes belonging to
/// one particle share the same ParticleSolidifierSnapshot.
///
/// ParticleBlobPartitioner is intentionally kept as the existing external
/// helper used by this project.
/// </summary>
public static class ParticleSolidifier
{
    private const float TerrainCurveStride = 0.06f;

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
        public object shapeCallbackTarget;
        public object bodyCallbackTarget;
        public int fluidTypeId;
        public float mass;
        public float density;
    }

    private sealed class CellGeometry
    {
        public readonly List<PolygonGeometry> polygons = new List<PolygonGeometry>();
        public int islandCount;
    }

    /// <summary>
    /// Partitions the supplied particles into proximity-connected blobs and
    /// solidifies each blob independently. Blobs with fewer than three valid
    /// particles are skipped.
    ///
    /// Once per partitioned blob, terrain candidates are discovered with one
    /// circle-overlap query filtered to terrainLayerName. The query circle
    /// encloses the complete generated Voronoi region. The name is resolved
    /// through PhysicsLayers. Pass null or an empty name for no terrain subtraction.
    /// </summary>
    public static List<PhysicsBody> Solidify(
        List<PhysicsBody> particles,
        float connectionDistance,
        float boundaryPadding = 0.25f,
        PhysicsBodyDefinition? solidBodyDefinition = null,
        string terrainLayerName = null)
    {
        return SolidifyGroupsInternal(
            particles != null ? particles.ToArray() : null,
            connectionDistance,
            boundaryPadding,
            solidBodyDefinition,
            null,
            terrainLayerName);
    }

    /// <summary>
    /// Same as Solidify, but also removes successfully solidified particles from
    /// the supplied PBF simulation. Liquify restores them to that simulation.
    /// </summary>
    public static List<PhysicsBody> Solidify(
        List<PhysicsBody> particles,
        PbfFluidLite simulation,
        float connectionDistance,
        float boundaryPadding = 0.25f,
        PhysicsBodyDefinition? solidBodyDefinition = null,
        string terrainLayerName = null)
    {
        return SolidifyGroupsInternal(
            particles != null ? particles.ToArray() : null,
            connectionDistance,
            boundaryPadding,
            solidBodyDefinition,
            simulation,
            terrainLayerName);
    }

    private static List<PhysicsBody> SolidifyGroupsInternal(
        PhysicsBody[] particles,
        float connectionDistance,
        float boundaryPadding,
        PhysicsBodyDefinition? solidBodyDefinition,
        PbfFluidLite simulation,
        string terrainLayerName)
    {
        PhysicsMask terrainLayerMask = PhysicsMask.None;

        if (!string.IsNullOrWhiteSpace(terrainLayerName))
        {
            terrainLayerMask = PhysicsLayers.GetLayerMask(terrainLayerName);

            if (!terrainLayerMask.AreAnyBitsSet(PhysicsMask.All))
            {
                Debug.LogError(
                    "[ParticleSolidifier] Terrain layer '" + terrainLayerName +
                    "' could not be resolved. Solidification was cancelled.");
                return new List<PhysicsBody>();
            }
        }

        List<List<PhysicsBody>> blobs = ParticleBlobPartitioner.PartitionBlobs(
            particles,
            connectionDistance);

        List<PhysicsBody> solidifiedBodies = new List<PhysicsBody>(blobs.Count);

        for (int i = 0; i < blobs.Count; i++)
        {
            if (blobs[i].Count < 3)
            {
                Debug.LogWarning(
                    "[ParticleSolidifier] Blob " + i + " contains " + blobs[i].Count +
                    " particles. At least 3 are required, so this blob was left unchanged.");
                continue;
            }

            PhysicsBody solidBody = SolidifyInternal(
                blobs[i],
                boundaryPadding,
                solidBodyDefinition,
                simulation,
                terrainLayerMask);

            if (solidBody.isValid)
                solidifiedBodies.Add(solidBody);
        }

        return solidifiedBodies;
    }

    /// <summary>
    /// Solidifies one already-partitioned blob. Each input body must have
    /// exactly one circular shape.
    /// </summary>
    private static PhysicsBody SolidifyInternal(
        List<PhysicsBody> particles,
        float boundaryPadding,
        PhysicsBodyDefinition? solidBodyDefinition,
        PbfFluidLite simulation,
        PhysicsMask terrainLayerMask)
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
            input.bodyCallbackTarget = particle.callbackTarget;

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
                input.shapeCallbackTarget = shape.callbackTarget;
            }

            input.fluidTypeId = simulation != null
                ? simulation.GetParticleType(particle)
                : -1;

            input.mass = particle.mass;
            input.density = input.shapeDefinition.density;
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

        HashSet<PhysicsBody> sourceBodies = new HashSet<PhysicsBody>();

        for (int i = 0; i < inputs.Count; i++)
        {
            positions.Add(inputs[i].position);
            center += inputs[i].position;
            averageVelocity += inputs[i].body.linearVelocity;
            averageAngularVelocity += inputs[i].body.angularVelocity;
            sourceBodies.Add(inputs[i].body);
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

        PhysicsTransform solidTransform = new PhysicsTransform(center, PhysicsRotate.identity);
        HashSet<PhysicsShape> unsupportedShapesAlreadyReported = new HashSet<PhysicsShape>();

        // Discover terrain once for the entire blob. The circle encloses the
        // generated Voronoi region, including its expanded outer boundary.
        // Every particle cell then reuses this same terrain candidate list.
        List<PhysicsShape> terrainCandidates = new List<PhysicsShape>();

        // Build and validate every cell's output geometry before creating or
        // destroying any source body. A completely removed cell is represented
        // by an empty polygon list; its particle is retained in the manifest.
        CellGeometry[] cellGeometry = new CellGeometry[inputs.Count];
        int totalOutputShapes = 0;
        int totalIslands = 0;

        try
        {
            if (terrainLayerMask.AreAnyBitsSet(PhysicsMask.All))
            {
                CircleGeometry blobQueryCircle = CalculateEnclosingCircle(cells, center);
                terrainCandidates = GetTerrainCandidates(
                    world,
                    blobQueryCircle,
                    terrainLayerMask,
                    sourceBodies);
            }

            for (int i = 0; i < cells.Length; i++)
            {
                List<Vector2> localVertices = new List<Vector2>(cells[i].Count);

                for (int vertexIndex = 0; vertexIndex < cells[i].Count; vertexIndex++)
                    localVertices.Add(cells[i][vertexIndex] - center);

                PolygonGeometry baseGeometry = PolygonGeometry.Create(localVertices.ToArray());

                if (!baseGeometry.isValid)
                    throw new InvalidOperationException(
                        "Invalid Voronoi polygon generated for particle " + i + ".");

                CellGeometry generated = ComposeCellGeometry(
                    baseGeometry,
                    solidTransform,
                    terrainCandidates,
                    unsupportedShapesAlreadyReported);

                cellGeometry[i] = generated;
                totalOutputShapes += generated.polygons.Count;
                totalIslands += generated.islandCount;
            }
        }
        catch (Exception exception)
        {
            Debug.LogError("[ParticleSolidifier] Terrain/Voronoi geometry generation failed. Source particles were left untouched.");
            Debug.LogException(exception);
            return default;
        }

        PhysicsBodyDefinition bodyDefinition = solidBodyDefinition.HasValue
            ? solidBodyDefinition.Value
            : inputs[0].bodyDefinition;

        bodyDefinition.position = center;
        bodyDefinition.rotation = PhysicsRotate.identity;
        bodyDefinition.linearVelocity = averageVelocity;
        bodyDefinition.angularVelocity = averageAngularVelocity;

        PhysicsBody solidBody = default;
        ParticleSolidifierManifest manifest = null;
        List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>(inputs.Count);
        int createdShapeCount = 0;

        try
        {
            solidBody = world.CreateBody(bodyDefinition);

            if (!solidBody.isValid)
                throw new InvalidOperationException("Failed to create the solidified body.");

            manifest = ScriptableObject.CreateInstance<ParticleSolidifierManifest>();
            manifest.hideFlags = HideFlags.HideAndDontSave;

            PhysicsUserData compoundUserData = solidBody.userData;
            compoundUserData.objectValue = manifest;
            solidBody.userData = compoundUserData;

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
                snapshot.simulation = simulation;
                snapshot.fluidTypeId = input.fluidTypeId;
                snapshot.mass = input.mass;
                snapshot.density = input.density;

                snapshots.Add(snapshot);
                manifest.snapshots.Add(snapshot);

                PhysicsShapeDefinition polygonDefinition = input.shapeDefinition;
                polygonDefinition.startMassUpdate = false;

                // Every convex polygon fragment belonging to this original
                // particle receives the same snapshot reference.
                for (int polygonIndex = 0;
                    polygonIndex < cellGeometry[i].polygons.Count;
                    polygonIndex++)
                {
                    PhysicsShape solidShape = solidBody.CreateShape(
                        cellGeometry[i].polygons[polygonIndex],
                        polygonDefinition);

                    if (!solidShape.isValid)
                    {
                        throw new InvalidOperationException(
                            "Failed to create polygon shape " + polygonIndex +
                            " for source particle " + i + ".");
                    }

                    PhysicsUserData shapeUserData = solidShape.userData;
                    shapeUserData.objectValue = snapshot;
                    solidShape.userData = shapeUserData;
                    solidShape.callbackTarget = input.shapeCallbackTarget;
                    createdShapeCount++;
                }
            }

            // Skip mass recalculation for the edge case where terrain removed
            // every cell. The empty body still acts as the snapshot holder.
            if (createdShapeCount > 0)
                solidBody.ApplyMassFromShapes();
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            if (solidBody.isValid)
                solidBody.Destroy();

            DestroySnapshots(snapshots);
            DestroyManifest(manifest);
            return default;
        }

        // Commit only after the full replacement body and every snapshot are
        // complete. Remove registrations first, then destroy all source bodies
        // in one PhysicsCore2D batch call.
        List<int> removedFluidTypeIds = new List<int>(inputs.Count);
        List<bool> removedFromSimulation = new List<bool>(inputs.Count);

        try
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                bool shouldRestoreRegistration = simulation != null;
                removedFluidTypeIds.Add(inputs[i].fluidTypeId);
                removedFromSimulation.Add(shouldRestoreRegistration);

                if (shouldRestoreRegistration)
                    simulation.RemoveParticle(inputs[i].body);
            }
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            if (simulation != null)
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    if (i >= removedFromSimulation.Count || !removedFromSimulation[i] ||
                        !inputs[i].body.isValid)
                    {
                        continue;
                    }

                    try
                    {
                        simulation.AddParticle(inputs[i].body, removedFluidTypeIds[i]);
                    }
                    catch (Exception rollbackException)
                    {
                        Debug.LogException(rollbackException);
                    }
                }
            }

            if (solidBody.isValid)
                solidBody.Destroy();

            DestroySnapshots(snapshots);
            DestroyManifest(manifest);
            return default;
        }

        PhysicsBody[] sourceBodyArray = new PhysicsBody[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
            sourceBodyArray[i] = inputs[i].body;

        PhysicsWorld.DestroyBodyBatch(sourceBodyArray);

        Debug.Log(
            "[ParticleSolidifier] Solidified " + inputs.Count +
            " particles into one body using " + totalOutputShapes +
            " convex shapes across " + totalIslands + " geometry islands.");

        return solidBody;
    }

    private static CellGeometry ComposeCellGeometry(
        PolygonGeometry baseGeometry,
        PhysicsTransform solidTransform,
        List<PhysicsShape> terrainShapes,
        HashSet<PhysicsShape> unsupportedShapesAlreadyReported)
    {
        CellGeometry result = new CellGeometry();

        if (terrainShapes.Count == 0)
        {
            result.polygons.Add(baseGeometry);
            result.islandCount = 1;
            return result;
        }

        PhysicsComposer composer = PhysicsComposer.Create(Allocator.Temp);

        try
        {
            if (!composer.isValid)
                throw new InvalidOperationException("PhysicsComposer.Create returned an invalid composer.");

            composer.maxPolygonVertices = PhysicsConstants.MaxPolygonVertices;

            // The original cell is already expressed in prospective solid-body
            // local space, so its composer transform is identity.
            composer.AddLayer(
                baseGeometry,
                new PhysicsTransform(Vector2.zero, PhysicsRotate.identity),
                PhysicsComposer.Operation.OR,
                0);

            int order = 1;

            for (int i = 0; i < terrainShapes.Count; i++)
            {
                PhysicsShape terrainShape = terrainShapes[i];

                if (!IsComposerSupportedTerrainShape(terrainShape))
                {
                    if (unsupportedShapesAlreadyReported.Add(terrainShape))
                    {
                        Debug.LogWarning(
                            "[ParticleSolidifier] Terrain shape type " + terrainShape.shapeType +
                            " is not supported by PhysicsComposer.AddLayer(PhysicsShape). " +
                            "Only Circle, Capsule and Polygon shapes are subtracted. This terrain shape was skipped.");
                    }

                    continue;
                }

                // PhysicsShape.transform is the terrain body's world transform.
                // Convert it into the local space of the body being created.
                PhysicsTransform terrainLocalTransform =
                    solidTransform.InverseMultiplyTransform(terrainShape.transform);

                composer.AddLayer(
                    terrainShape,
                    terrainLocalTransform,
                    PhysicsComposer.Operation.NOT,
                    order,
                    TerrainCurveStride);

                order++;
            }

            NativeArray<PolygonGeometry> composedPolygons =
                composer.CreatePolygonGeometry(Allocator.Temp);

            try
            {
                NativeArray<RangeInt> islands = composer.GetGeometryIslands(Allocator.Temp);

                try
                {
                    result.islandCount = islands.Length;

                    if (composer.rejectedGeometryCount > 0)
                    {
                        throw new InvalidOperationException(
                            "PhysicsComposer rejected " + composer.rejectedGeometryCount +
                            " output geometry element(s). The cell was not solidified to avoid silently losing thin geometry.");
                    }

                    bool[] polygonAssignedToIsland = new bool[composedPolygons.Length];

                    for (int islandIndex = 0; islandIndex < islands.Length; islandIndex++)
                    {
                        RangeInt island = islands[islandIndex];

                        if (island.start < 0 || island.length < 0 ||
                            island.start + island.length > composedPolygons.Length)
                        {
                            throw new InvalidOperationException(
                                "PhysicsComposer returned an out-of-range geometry island.");
                        }

                        for (int islandOffset = 0; islandOffset < island.length; islandOffset++)
                        {
                            int polygonIndex = island.start + islandOffset;

                            if (polygonAssignedToIsland[polygonIndex])
                            {
                                throw new InvalidOperationException(
                                    "PhysicsComposer returned overlapping geometry island ranges.");
                            }

                            polygonAssignedToIsland[polygonIndex] = true;
                            PolygonGeometry polygon = composedPolygons[polygonIndex];

                            if (!polygon.isValid || polygon.count < 3 ||
                                polygon.count > PhysicsConstants.MaxPolygonVertices)
                            {
                                throw new InvalidOperationException(
                                    "PhysicsComposer returned an invalid convex polygon at index " + polygonIndex + ".");
                            }

                            // Retain polygons grouped by their returned island
                            // ranges. Each polygon becomes an individual convex
                            // PhysicsShape, while the source-particle snapshot
                            // remains shared by all of them.
                            result.polygons.Add(polygon);
                        }
                    }

                    for (int polygonIndex = 0; polygonIndex < polygonAssignedToIsland.Length; polygonIndex++)
                    {
                        if (!polygonAssignedToIsland[polygonIndex])
                        {
                            throw new InvalidOperationException(
                                "PhysicsComposer.GetGeometryIslands did not include output polygon " +
                                polygonIndex + ".");
                        }
                    }
                }
                finally
                {
                    if (islands.IsCreated)
                        islands.Dispose();
                }
            }
            finally
            {
                if (composedPolygons.IsCreated)
                    composedPolygons.Dispose();
            }
        }
        finally
        {
            composer.Destroy();
        }

        return result;
    }

    private static bool IsComposerSupportedTerrainShape(PhysicsShape shape)
    {
        return shape.shapeType == PhysicsShape.ShapeType.Circle ||
               shape.shapeType == PhysicsShape.ShapeType.Capsule ||
               shape.shapeType == PhysicsShape.ShapeType.Polygon;
    }

    private static List<PhysicsShape> GetTerrainCandidates(
        PhysicsWorld world,
        CircleGeometry worldCircle,
        PhysicsMask terrainLayerMask,
        HashSet<PhysicsBody> sourceBodies)
    {
        List<PhysicsShape> candidates = new List<PhysicsShape>();
        HashSet<PhysicsShape> uniqueShapes = new HashSet<PhysicsShape>();

        // This query is performed once for the whole blob. Its hitCategories
        // restrict results to the requested terrain layer; PhysicsMask.All
        // allows the query to hit terrain regardless of contact masks.
        PhysicsQuery.QueryFilter queryFilter = new PhysicsQuery.QueryFilter(
            PhysicsMask.All,
            terrainLayerMask);

        NativeArray<PhysicsQuery.WorldOverlapResult> overlaps = world.OverlapGeometry(
            worldCircle,
            queryFilter,
            Allocator.Temp);

        try
        {
            for (int i = 0; i < overlaps.Length; i++)
            {
                if (!overlaps[i].isValid)
                    continue;

                PhysicsShape shape = overlaps[i].shape;

                if (!shape.isValid || !shape.world.Equals(world))
                    continue;

                // The source particles are never terrain, even if their
                // PhysicsCore2D category happens to include terrainLayerMask.
                if (sourceBodies.Contains(shape.body))
                    continue;

                if (uniqueShapes.Add(shape))
                    candidates.Add(shape);
            }
        }
        finally
        {
            if (overlaps.IsCreated)
                overlaps.Dispose();
        }

        return candidates;
    }

    private static CircleGeometry CalculateEnclosingCircle(
        List<Vector2>[] worldCells,
        Vector2 blobCenter)
    {
        // The circle is centered on the blob center. Its radius reaches the
        // farthest generated Voronoi vertex, which also covers the farthest
        // particle and the expansion applied to the outer Voronoi hull.
        float radiusSquared = 0f;

        for (int cellIndex = 0; cellIndex < worldCells.Length; cellIndex++)
        {
            List<Vector2> cell = worldCells[cellIndex];

            for (int vertexIndex = 0; vertexIndex < cell.Count; vertexIndex++)
            {
                float distanceSquared = (cell[vertexIndex] - blobCenter).sqrMagnitude;
                radiusSquared = Mathf.Max(radiusSquared, distanceSquared);
            }
        }

        // This is one broadphase overlap query for the whole blob, not a query
        // per particle. The small margin protects against boundary precision;
        // PhysicsComposer performs the exact per-cell subtraction afterwards.
        float radius = Mathf.Sqrt(radiusSquared) + 0.01f;
        return CircleGeometry.Create(radius, blobCenter);
    }

    /// <summary>
    /// Recreates every original circular particle exactly once, even when its
    /// cell generated several convex shapes or no shapes at all. Newly created
    /// particles follow the solid body's current pose and point velocity.
    /// </summary>
    public static List<PhysicsBody> Liquify(PhysicsBody solidifiedBody)
    {
        List<PhysicsBody> particles = new List<PhysicsBody>();

        if (!solidifiedBody.isValid)
        {
            Debug.LogWarning("[ParticleSolidifier] Cannot liquify an invalid body.");
            return particles;
        }

        ParticleSolidifierManifest manifest;
        bool manifestIncomplete;
        List<ParticleSolidifierSnapshot> snapshots =
            CollectSnapshots(solidifiedBody, out manifest, out manifestIncomplete);

        if (manifestIncomplete)
        {
            Debug.LogError(
                "[ParticleSolidifier] At least one particle snapshot referenced by the body manifest is missing. " +
                "The body was retained because restoring it would silently lose an original particle.");
            return particles;
        }

        if (snapshots.Count == 0)
        {
            Debug.LogWarning(
                "[ParticleSolidifier] No particle snapshots were found. The solidified body was not destroyed.");
            return particles;
        }

        List<bool> registeredInSimulation = new List<bool>(snapshots.Count);

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

                // Add it immediately so any subsequent failure is covered by
                // the rollback path below.
                particles.Add(particle);
                registeredInSimulation.Add(false);

                particle.userData = snapshot.bodyUserData;
                particle.callbackTarget = snapshot.bodyCallbackTarget;

                PhysicsShapeDefinition shapeDefinition = snapshot.shapeDefinition;
                shapeDefinition.density = snapshot.density;

                PhysicsShape particleShape = particle.CreateShape(
                    snapshot.circleGeometry,
                    shapeDefinition);

                if (!particleShape.isValid)
                    throw new InvalidOperationException("Failed to recreate particle shape " + i + ".");

                particleShape.userData = snapshot.shapeUserData;
                particleShape.callbackTarget = snapshot.shapeCallbackTarget;
                if (snapshot.mass > 1e-6f)
                    particle.mass = snapshot.mass;

                if (snapshot.simulation != null)
                {
                    registeredInSimulation[i] = true;
                    snapshot.simulation.AddParticle(particle, snapshot.fluidTypeId);
                }
            }

            // Do not discard the snapshots unless the old compound body is
            // actually destroyed. This leaves a retryable state on failure.
            if (!solidifiedBody.Destroy())
                throw new InvalidOperationException("Failed to destroy the solidified body after reconstruction.");
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);

            for (int i = 0; i < particles.Count; i++)
            {
                if (registeredInSimulation[i] && snapshots[i].simulation != null)
                {
                    try
                    {
                        snapshots[i].simulation.RemoveParticle(particles[i]);
                    }
                    catch (Exception rollbackException)
                    {
                        Debug.LogException(rollbackException);
                    }
                }

                if (particles[i].isValid)
                {
                    try
                    {
                        particles[i].Destroy();
                    }
                    catch (Exception rollbackException)
                    {
                        Debug.LogException(rollbackException);
                    }
                }
            }

            // Keep the solid body and its snapshots intact so the operation can
            // be retried or inspected after a failed reconstruction.
            return new List<PhysicsBody>();
        }

        DestroySnapshots(snapshots);
        DestroyManifest(manifest);

        Debug.Log("[ParticleSolidifier] Liquified " + particles.Count + " original particles.");
        return particles;
    }

    /// <summary>
    /// Destroys a solidified body and its snapshots without liquifying it.
    /// Use this instead of PhysicsBody.Destroy when discarding a solidified body.
    /// The body-level manifest is used so snapshots for fully-subtracted cells
    /// are also destroyed.
    /// </summary>
    public static void DestroySolidifiedBody(PhysicsBody solidifiedBody)
    {
        if (!solidifiedBody.isValid)
            return;

        ParticleSolidifierManifest manifest;
        bool manifestIncomplete;
        List<ParticleSolidifierSnapshot> snapshots =
            CollectSnapshots(solidifiedBody, out manifest, out manifestIncomplete);

        if (manifestIncomplete)
        {
            Debug.LogWarning(
                "[ParticleSolidifier] The manifest contains missing snapshots. The body will be destroyed, " +
                "but already-missing source particles cannot be recovered.");
        }

        if (!solidifiedBody.Destroy())
        {
            Debug.LogWarning("[ParticleSolidifier] Could not destroy the solidified body; its snapshots were retained.");
            return;
        }

        DestroySnapshots(snapshots);
        DestroyManifest(manifest);
    }

    private static List<ParticleSolidifierSnapshot> CollectSnapshots(
        PhysicsBody solidifiedBody,
        out ParticleSolidifierManifest manifest,
        out bool manifestIncomplete)
    {
        manifestIncomplete = false;
        List<ParticleSolidifierSnapshot> snapshots = new List<ParticleSolidifierSnapshot>();
        HashSet<ParticleSolidifierSnapshot> uniqueSnapshots =
            new HashSet<ParticleSolidifierSnapshot>();

        PhysicsUserData bodyUserData = solidifiedBody.userData;
        manifest = bodyUserData.objectValue as ParticleSolidifierManifest;

        if (manifest != null)
        {
            if (manifest.snapshots == null)
            {
                manifestIncomplete = true;
                Debug.LogError(
                    "[ParticleSolidifier] The body manifest snapshot list is missing.");
            }
            else
            {
                for (int i = 0; i < manifest.snapshots.Count; i++)
                {
                    ParticleSolidifierSnapshot snapshot = manifest.snapshots[i];

                    if (snapshot == null)
                    {
                        manifestIncomplete = true;
                        Debug.LogError(
                            "[ParticleSolidifier] The body manifest contains a missing snapshot at index " + i + ".");
                        continue;
                    }

                    if (uniqueSnapshots.Add(snapshot))
                        snapshots.Add(snapshot);
                }
            }
        }

        // Also inspect shape user data. This deduplicates fragments belonging
        // to the same source particle and supports solid bodies created by the
        // previous one-shape-per-particle implementation (without a manifest).
        using (NativeArray<PhysicsShape> shapes = solidifiedBody.GetShapes())
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                PhysicsUserData userData = shapes[i].userData;
                ParticleSolidifierSnapshot snapshot =
                    userData.objectValue as ParticleSolidifierSnapshot;

                if (snapshot != null && uniqueSnapshots.Add(snapshot))
                    snapshots.Add(snapshot);
            }
        }

        return snapshots;
    }

    private static void DestroySnapshots(List<ParticleSolidifierSnapshot> snapshots)
    {
        if (snapshots == null)
            return;

        HashSet<ParticleSolidifierSnapshot> uniqueSnapshots =
            new HashSet<ParticleSolidifierSnapshot>();

        for (int i = 0; i < snapshots.Count; i++)
        {
            ParticleSolidifierSnapshot snapshot = snapshots[i];

            if (snapshot != null && uniqueSnapshots.Add(snapshot))
                UnityEngine.Object.Destroy(snapshot);
        }
    }

    private static void DestroyManifest(ParticleSolidifierManifest manifest)
    {
        if (manifest != null)
            UnityEngine.Object.Destroy(manifest);
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
