using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(BMeshTerrain2D))]
public class BMeshTerrain2DEditor : Editor
{
    private BMeshTerrain2D terrain;

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

        if (GUILayout.Button("Regenerate Terrain"))
        {
            Undo.RecordObject(terrain, "Regenerate Terrain");
            terrain.Regenerate();
            EditorUtility.SetDirty(terrain);
            PrefabUtility.RecordPrefabInstancePropertyModifications(terrain);
        }
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

        for (int i = 0; i < count; i++)
        {
            Vector3 worldPosition = worldPoints[i];
            float handleSize = HandleUtility.GetHandleSize(worldPosition);

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
