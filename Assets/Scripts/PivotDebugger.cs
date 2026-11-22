using UnityEngine;

public class PivotDebugger : MonoBehaviour
{
    public void OnDrawGizmos()
    {
        Gizmos.color = Color.lawnGreen;
        Gizmos.DrawSphere(transform.position,0.25f);

        Gizmos.color = Color.crimson;
        foreach (Transform item in transform.GetComponentsInChildren<Transform>())
        {
            Gizmos.DrawSphere(item.position,0.2f);
        }
    }
}
