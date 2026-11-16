using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.U2D.Physics;

/// <summary>
/// Splits a supplied collection of particle bodies into proximity-connected blobs.
/// All spatial data is built locally from the supplied particles on each call.
/// </summary>
public static class ParticleBlobPartitioner
{
    private struct CellKey : IEquatable<CellKey>
    {
        public int worldIndex;
        public Vector2Int cell;

        public CellKey(int worldIndex, Vector2Int cell)
        {
            this.worldIndex = worldIndex;
            this.cell = cell;
        }

        public bool Equals(CellKey other)
        {
            return worldIndex == other.worldIndex && cell == other.cell;
        }

        public override bool Equals(object obj)
        {
            return obj is CellKey && Equals((CellKey)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (worldIndex * 397) ^ cell.GetHashCode();
            }
        }
    }

    private struct ExactPositionKey : IEquatable<ExactPositionKey>
    {
        public int worldIndex;
        public Vector2 position;

        public ExactPositionKey(int worldIndex, Vector2 position)
        {
            this.worldIndex = worldIndex;
            this.position = position;
        }

        public bool Equals(ExactPositionKey other)
        {
            return worldIndex == other.worldIndex && position == other.position;
        }

        public override bool Equals(object obj)
        {
            return obj is ExactPositionKey && Equals((ExactPositionKey)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (worldIndex * 397) ^ position.GetHashCode();
            }
        }
    }

    /// <summary>
    /// Returns connected groups of valid, unique particle bodies. Two particles
    /// are directly connected when their world-space centres are no farther apart
    /// than range. Connectivity is transitive.
    /// Bodies from different PhysicsWorld instances are kept in separate blobs.
    /// </summary>
    public static List<List<PhysicsBody>> PartitionBlobs(PhysicsBody[] particles, float range)
    {
        if (float.IsNaN(range) || float.IsInfinity(range) || range < 0f)
            throw new ArgumentOutOfRangeException(nameof(range), "Range must be finite and non-negative.");

        List<List<PhysicsBody>> result = new List<List<PhysicsBody>>();
        if (particles == null || particles.Length == 0)
            return result;

        List<PhysicsBody> validBodies = new List<PhysicsBody>(particles.Length);
        List<Vector2> positions = new List<Vector2>(particles.Length);
        List<int> worldIndices = new List<int>(particles.Length);
        List<PhysicsWorld> worlds = new List<PhysicsWorld>();
        HashSet<string> seenHandles = new HashSet<string>();

        // These dictionaries exist only for this invocation and contain only
        // particles supplied by the caller. They are never shared with a simulation.
        Dictionary<CellKey, List<int>> spatialGrid = new Dictionary<CellKey, List<int>>();
        Dictionary<ExactPositionKey, List<int>> exactPositionGrid = new Dictionary<ExactPositionKey, List<int>>();

        for (int i = 0; i < particles.Length; i++)
        {
            PhysicsBody body = particles[i];
            if (!body.isValid)
                continue;

            int worldIndex = GetWorldIndex(worlds, body.world);
            PhysicsHandle handle = body.physicsHandle;
            string handleKey = worldIndex + ":" + handle.generation + ":" + handle.index;
            if (!seenHandles.Add(handleKey))
                continue;

            int bodyIndex = validBodies.Count;
            Vector2 position = body.position;
            validBodies.Add(body);
            positions.Add(position);
            worldIndices.Add(worldIndex);

            if (range == 0f)
            {
                ExactPositionKey key = new ExactPositionKey(worldIndex, position);
                AddIndex(exactPositionGrid, key, bodyIndex);
            }
            else
            {
                CellKey key = new CellKey(worldIndex, GetCell(position, range));
                AddIndex(spatialGrid, key, bodyIndex);
            }
        }

        bool[] visited = new bool[validBodies.Count];
        float rangeSquared = range * range;
        Queue<int> queue = new Queue<int>();

        for (int startIndex = 0; startIndex < validBodies.Count; startIndex++)
        {
            if (visited[startIndex])
                continue;

            List<int> blobIndices = new List<int>();
            queue.Clear();
            queue.Enqueue(startIndex);
            visited[startIndex] = true;

            while (queue.Count > 0)
            {
                int currentIndex = queue.Dequeue();
                blobIndices.Add(currentIndex);

                if (range == 0f)
                {
                    ExactPositionKey exactKey = new ExactPositionKey(
                        worldIndices[currentIndex], positions[currentIndex]);
                    List<int> matches;
                    if (!exactPositionGrid.TryGetValue(exactKey, out matches))
                        continue;

                    for (int i = 0; i < matches.Count; i++)
                    {
                        int candidateIndex = matches[i];
                        if (!visited[candidateIndex])
                        {
                            visited[candidateIndex] = true;
                            queue.Enqueue(candidateIndex);
                        }
                    }

                    continue;
                }

                Vector2 currentPosition = positions[currentIndex];
                Vector2Int currentCell = GetCell(currentPosition, range);
                int currentWorldIndex = worldIndices[currentIndex];

                for (int offsetX = -1; offsetX <= 1; offsetX++)
                {
                    for (int offsetY = -1; offsetY <= 1; offsetY++)
                    {
                        CellKey key = new CellKey(
                            currentWorldIndex,
                            new Vector2Int(currentCell.x + offsetX, currentCell.y + offsetY));
                        List<int> candidates;
                        if (!spatialGrid.TryGetValue(key, out candidates))
                            continue;

                        for (int i = 0; i < candidates.Count; i++)
                        {
                            int candidateIndex = candidates[i];
                            if (visited[candidateIndex])
                                continue;

                            if ((positions[candidateIndex] - currentPosition).sqrMagnitude <= rangeSquared)
                            {
                                visited[candidateIndex] = true;
                                queue.Enqueue(candidateIndex);
                            }
                        }
                    }
                }
            }

            blobIndices.Sort();
            List<PhysicsBody> blob = new List<PhysicsBody>(blobIndices.Count);
            for (int i = 0; i < blobIndices.Count; i++)
                blob.Add(validBodies[blobIndices[i]]);

            result.Add(blob);
        }

        return result;
    }

    private static int GetWorldIndex(List<PhysicsWorld> worlds, PhysicsWorld world)
    {
        for (int i = 0; i < worlds.Count; i++)
        {
            if (worlds[i].Equals(world))
                return i;
        }

        worlds.Add(world);
        return worlds.Count - 1;
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
