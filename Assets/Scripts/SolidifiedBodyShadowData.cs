using UnityEngine;

/// <summary>
/// Data source used by SolidifiedParticleShadowProvider.
/// Boundary vertices and line indices describe the outer and inner contours
/// of the solidified body's polygon union in local space.
/// </summary>
[DisallowMultipleComponent]
public sealed class SolidifiedBodyShadowData : MonoBehaviour
{
    [SerializeField, HideInInspector] private Vector2[] boundaryVertices = new Vector2[0];
    [SerializeField, HideInInspector] private int[] boundaryIndices = new int[0];

    public Vector2[] BoundaryVertices => boundaryVertices;
    public int[] BoundaryIndices => boundaryIndices;

    public void SetBoundaryShape(Vector2[] vertices, int[] indices)
    {
        boundaryVertices = vertices != null ? vertices : new Vector2[0];
        boundaryIndices = indices != null ? indices : new int[0];
    }
}
