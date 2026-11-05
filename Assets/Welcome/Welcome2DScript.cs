using UnityEngine;
using UnityEditor;

namespace Unity.U2D.Welcome
{
    [HelpURL("https://unity.com/features/2d")]
    class Welcome2DScript : ScriptableObject
    {
        public static void Open2DLandingPage()
        {
            Application.OpenURL("https://unity.com/features/2d");
        }

        // Disable tutorial menu item since 2D doesn't provide one.
        [MenuItem("Tutorials/Show Tutorials Window", true)]
        static bool ValidateShowTutorials() => false;
    }


}
