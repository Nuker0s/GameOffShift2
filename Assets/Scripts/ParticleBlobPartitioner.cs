using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Partitions particle bodies into connected components based on the distance
/// between their world-space centres. Reusable outside ParticleSolidifier.
/// </summary>
public static class ParticleBlobPartitioner
{
    /// <summary>
    /// Finds connected groups of particle bodies using their world-space positions.
    /// Two particles are directly connected when their centres are no farther apart
    /// than connectionDistance. Connectivity is transitive: A-B-C is one blob when
    /// A is connected to B and B is connected to C, even if A and C are farther apart.
    /// Invalid bodies and duplicate body handles are ignored. Bodies from different
    /// PhysicsWorld instances are never placed in the same blob.
    /// </summary>
    public static PhysicsBody[][] GetBlobs(PhysicsBody[] particles, float connectionDistance)
    {
        if (float.IsNaN(connectionDistance) || float.IsInfinity(connectionDistance) || connectionDistance < 0f)
            throw new ArgumentOutOfRangeException(nameof(connectionDistance), "Connection distance must be a finite, non-negative value.");

        if (particles == null || particles.Length == 0)
            return new PhysicsBody[0][];

        List<PhysicsBody> validBodies = new List<PhysicsBody>(particles.Length);
        List<Vector2> positions = new List<Vector2>(particles.Length);
        List<int> particleWorldGridIndices = new List<int>(particles.Length);
        List<WorldGrid> worldGrids = new List<WorldGrid>();

        for (int i = 0; i < particles.Length; i++)
        {
            PhysicsBody body = particles[i];

            if (!body.isValid)
                continue;

            int worldGridIndex = FindOrCreateWorldGrid(worldGrids, body.world);
            WorldGrid worldGrid = worldGrids[worldGridIndex];

            // A body handle is unique within its world. Keep the world grids
            // separate because handle indices can overlap between worlds.
            if (!worldGrid.bodyHandles.Add(GetHandleKey(body.physicsHandle)))
                continue;

            Vector2 position = body.position;
            int bodyIndex = validBodies.Count;

            validBodies.Add(body);
            positions.Add(position);
            particleWorldGridIndices.Add(worldGridIndex);

            if (connectionDistance == 0f)
            {
                AddIndex(worldGrid.exactPositions, position, bodyIndex);
            }
            else
            {
                Vector2Int cell = GetCell(position, connectionDistance);
                AddIndex(worldGrid.cells, cell, bodyIndex);
            }
        }

        if (validBodies.Count == 0)
            return new PhysicsBody[0][];

        List<PhysicsBody[]> blobs = new List<PhysicsBody[]>();
        bool[] visited = new bool[validBodies.Count];

        // With a zero threshold, only particles at exactly the same position
        // are connected. Handle this separately because a zero-sized grid cell
        // would not be meaningful.
        if (connectionDistance == 0f)
        {
            for (int i = 0; i < validBodies.Count; i++)
            {
                if (visited[i])
                    continue;

                WorldGrid worldGrid = worldGrids[particleWorldGridIndices[i]];
                List<int> matchingIndices;

                if (!worldGrid.exactPositions.TryGetValue(positions[i], out matchingIndices))
                    continue;

                List<PhysicsBody> blob = new List<PhysicsBody>(matchingIndices.Count);

                for (int j = 0; j < matchingIndices.Count; j++)
                {
                    int matchingIndex = matchingIndices[j];

                    if (visited[matchingIndex])
                        continue;

                    visited[matchingIndex] = true;
                    blob.Add(validBodies[matchingIndex]);
                }

                if (blob.Count > 0)
                    blobs.Add(blob.ToArray());
            }

            return blobs.ToArray();
        }

        float connectionDistanceSquared = connectionDistance * connectionDistance;
        Queue<int> searchQueue = new Queue<int>();

        // Flood-fill the proximity graph to find its connected components.
        for (int startIndex = 0; startIndex < validBodies.Count; startIndex++)
        {
            if (visited[startIndex])
                continue;

            List<int> blobIndices = new List<int>();
            searchQueue.Clear();
            searchQueue.Enqueue(startIndex);
            visited[startIndex] = true;

            while (searchQueue.Count > 0)
            {
                int currentIndex = searchQueue.Dequeue();
                Vector2 currentPosition = positions[currentIndex];
                WorldGrid worldGrid = worldGrids[particleWorldGridIndices[currentIndex]];
                Vector2Int currentCell = GetCell(currentPosition, connectionDistance);

                blobIndices.Add(currentIndex);

                // With cell size equal to connectionDistance, every point
                // within range must be in this cell or one of its 8 neighbours.
                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    for (int offsetY = -1; offsetY <= 1; offsetY++)
                    {
                        Vector2Int neighbourCell = new Vector2Int(
                            currentCell.x + offsetX,
                            currentCell.y + offsetY);

                        List<int> candidates;

                        if (!worldGrid.cells.TryGetValue(neighbourCell, out candidates))
                            continue;

                        for (int candidateIndex = 0; candidateIndex < candidates.Count; candidateIndex++)
                        {
                            int neighbourIndex = candidates[candidateIndex];

                            if (visited[neighbourIndex])
                                continue;

                            Vector2 difference = positions[neighbourIndex] - currentPosition;

                            if (difference.sqrMagnitude <= connectionDistanceSquared)
                            {
                                visited[neighbourIndex] = true;
                                searchQueue.Enqueue(neighbourIndex);
                            }
                        }
                    }
                }
            }

            // Keep members in the same relative order as the input array,
            // regardless of the order in which the flood-fill discovered them.
            blobIndices.Sort();
            PhysicsBody[] blob = new PhysicsBody[blobIndices.Count];

            for (int i = 0; i < blobIndices.Count; i++)
                blob[i] = validBodies[blobIndices[i]];

            blobs.Add(blob);
        }

        return blobs.ToArray();
    }

    private sealed class WorldGrid
    {
        public PhysicsWorld world;
        public Dictionary<Vector2Int, List<int>> cells = new Dictionary<Vector2Int, List<int>>();
        public Dictionary<Vector2, List<int>> exactPositions = new Dictionary<Vector2, List<int>>();
        public HashSet<long> bodyHandles = new HashSet<long>();

        public WorldGrid(PhysicsWorld world)
        {
            this.world = world;
        }
    }

    private static int FindOrCreateWorldGrid(List<WorldGrid> worldGrids, PhysicsWorld world)
    {
        for (int i = 0; i < worldGrids.Count; i++)
        {
            if (worldGrids[i].world.Equals(world))
                return i;
        }

        worldGrids.Add(new WorldGrid(world));
        return worldGrids.Count - 1;
    }

    private static long GetHandleKey(PhysicsHandle handle)
    {
        return ((long)handle.generation << 32) | (uint)handle.index;
    }

    private static Vector2Int GetCell(Vector2 position, float cellSize)
    {
        return new Vector2Int(
            Mathf.FloorToInt(position.x / cellSize),
            Mathf.FloorToInt(position.y / cellSize));
    }

    private static void AddIndex<TKey>(Dictionary<TKey, List<int>> dictionary, TKey key, int index)
    {
        List<int> indices;

        if (!dictionary.TryGetValue(key, out indices))
        {
            indices = new List<int>();
            dictionary.Add(key, indices);
        }

        indices.Add(index);
    }

}
