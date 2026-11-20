using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(BMeshTerrain2D))]
public class BMeshTerrain2DEditor : Editor
{
    private BMeshTerrain2D terrain;
    private int selectedEdgeToSplit;

    private void OnEnable()
    {
        terrain = (BMeshTerrain2D)target;
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        EditorGUILayout.Space(8f);

        EditorGUILayout.BeginHorizontal();

        if (GUILayout.Button("Add Point"))
        {
            Undo.RecordObject(terrain, "Add Terrain Point");

            int count = terrain.outlinePoints.Count;

            if (count >= 2)
            {
                Vector3 newPoint = Vector3.Lerp(
                    terrain.outlinePoints[count - 1],
                    terrain.outlinePoints[0],
                    0.5f
                );

                terrain.outlinePoints.Add(newPoint);
            }
            else
            {
                terrain.outlinePoints.Add(Vector3.zero);
            }

            EditorUtility.SetDirty(terrain);
        }

        if (GUILayout.Button("Remove Last Point"))
        {
            if (terrain.outlinePoints.Count > 0)
            {
                Undo.RecordObject(terrain, "Remove Terrain Point");
                terrain.outlinePoints.RemoveAt(terrain.outlinePoints.Count - 1);
                EditorUtility.SetDirty(terrain);
            }
        }

        EditorGUILayout.EndHorizontal();

        EditorGUILayout.Space(4f);

        int pointCount = terrain.outlinePoints.Count;

        if (pointCount >= 2)
        {
            EditorGUILayout.LabelField("Split Edge", EditorStyles.boldLabel);
            EditorGUILayout.HelpBox(
                "Tip: you can also click a small sphere in the middle of any edge in the Scene view to split it.",
                MessageType.None
            );

            EditorGUILayout.BeginHorizontal();

            selectedEdgeToSplit = EditorGUILayout.IntField(
                "Edge (i -> i+1)",
                selectedEdgeToSplit
            );
            selectedEdgeToSplit = Mathf.Clamp(
                selectedEdgeToSplit,
                0,
                Mathf.Max(0, pointCount - 1)
            );

            EditorGUILayout.LabelField(
                string.Format(
                    "({0} -> {1})",
                    selectedEdgeToSplit,
                    (selectedEdgeToSplit + 1) % Mathf.Max(1, pointCount)
                ),
                GUILayout.Width(70f)
            );

            if (GUILayout.Button("Split Edge", GUILayout.Width(90f)))
            {
                SplitEdge(selectedEdgeToSplit);
            }

            EditorGUILayout.EndHorizontal();
        }

        if (GUILayout.Button("Regenerate Terrain"))
        {
            Undo.RecordObject(terrain, "Regenerate Terrain");
            terrain.Regenerate();
            EditorUtility.SetDirty(terrain);
            PrefabUtility.RecordPrefabInstancePropertyModifications(terrain);
        }
    }

    private void SplitEdge(int edgeIndex)
    {
        if (terrain == null || terrain.outlinePoints == null)
            return;

        int count = terrain.outlinePoints.Count;

        if (count < 2)
            return;

        edgeIndex = ((edgeIndex % count) + count) % count;
        int nextIndex = (edgeIndex + 1) % count;

        Vector3 midPoint = Vector3.Lerp(
            terrain.outlinePoints[edgeIndex],
            terrain.outlinePoints[nextIndex],
            0.5f
        );

        Undo.RecordObject(terrain, "Split Terrain Edge");
        terrain.outlinePoints.Insert(edgeIndex + 1, midPoint);

        // Keep the selection pointing at the second half of the split edge
        // so repeated splits keep subdividing the same area.
        selectedEdgeToSplit = edgeIndex + 1;

        terrain.Regenerate();

        EditorUtility.SetDirty(terrain);
        PrefabUtility.RecordPrefabInstancePropertyModifications(terrain);
    }

    private void OnSceneGUI()
    {
        if (Application.isPlaying || terrain == null || terrain.outlinePoints == null)
            return;

        int count = terrain.outlinePoints.Count;

        if (count == 0)
            return;

        Vector3[] worldPoints = new Vector3[count + 1];

        for (int i = 0; i < count; i++)
        {
            worldPoints[i] = terrain.transform.TransformPoint(
                terrain.outlinePoints[i]
            );
        }

        worldPoints[count] = worldPoints[0];

        Color previousColor = Handles.color;
        Handles.color = Color.cyan;
        Handles.DrawAAPolyLine(3f, worldPoints);

        // Highlight the edge selected in the Inspector.
        if (count >= 2)
        {
            selectedEdgeToSplit = Mathf.Clamp(selectedEdgeToSplit, 0, count - 1);
            int nextSelected = (selectedEdgeToSplit + 1) % count;
            Handles.color = Color.yellow;
            Handles.DrawAAPolyLine(
                5f,
                worldPoints[selectedEdgeToSplit],
                worldPoints[nextSelected]
            );
        }

        // Clickable midpoint handles: click one to split that edge.
        if (count >= 2)
        {
            for (int i = 0; i < count; i++)
            {
                int nextIndex = (i + 1) % count;
                Vector3 midWorld = Vector3.Lerp(
                    worldPoints[i],
                    worldPoints[nextIndex],
                    0.5f
                );

                float midHandleSize = HandleUtility.GetHandleSize(midWorld);
                float buttonSize = midHandleSize * 0.12f;
                float pickSize = midHandleSize * 0.18f;

                bool isSelected = (i == selectedEdgeToSplit);
                Handles.color = isSelected ? Color.yellow : Color.white;

                // Hovering highlights via handle color is handled by Unity;
                // keep the dot small so it doesn't obscure the edge.
                if (Handles.Button(
                    midWorld,
                    Quaternion.identity,
                    buttonSize,
                    pickSize,
                    Handles.SphereHandleCap
                ))
                {
                    selectedEdgeToSplit = i;
                    SplitEdge(i);
                    // Points changed - bail out so we don't draw stale handles
                    // this frame.
                    Handles.color = previousColor;
                    return;
                }

                // Small "+" label so users discover the split affordance.
                Handles.Label(midWorld, "+");
            }
        }

        for (int i = 0; i < count; i++)
        {
            Vector3 worldPosition = worldPoints[i];
            float handleSize = HandleUtility.GetHandleSize(worldPosition);

            Handles.color = previousColor;
            Handles.Label(
                worldPosition + Vector3.up * handleSize * 0.15f,
                i.ToString()
            );

            EditorGUI.BeginChangeCheck();

            Vector3 newWorldPosition = Handles.PositionHandle(
                worldPosition,
                terrain.transform.rotation
            );

            if (EditorGUI.EndChangeCheck())
            {
                Undo.RecordObject(terrain, "Move Terrain Vertex");

                terrain.outlinePoints[i] =
                    terrain.transform.InverseTransformPoint(newWorldPosition);

                terrain.Regenerate();

                EditorUtility.SetDirty(terrain);
                PrefabUtility.RecordPrefabInstancePropertyModifications(terrain);
            }
        }

        Handles.color = previousColor;
    }
}
