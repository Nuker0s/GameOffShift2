using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Drag and drop FPS counter.
/// Just drop this on any GameObject in the scene - it automatically
/// spawns its own Canvas + Text in the top-right and scales with screen size.
/// Uses legacy uGUI (UnityEngine.UI), no setup required.
/// </summary>
public class FPSCounter : MonoBehaviour
{
    [Header("Appearance")]
    [Tooltip("Base font size at 1920x1080 reference. Scales automatically with screen size via CanvasScaler.")]
    public int fontSize = 24;

    [Tooltip("Distance from top-right corner in pixels (at reference resolution).")]
    public Vector2 padding = new Vector2(15f, 15f);

    [Tooltip("Size of the text box.")]
    public Vector2 boxSize = new Vector2(300f, 50f);

    public Color textColor = Color.white;

    [Header("Update")]
    [Tooltip("How often to refresh the displayed number. Smaller = more flickery.")]
    public float updateInterval = 0.5f;

    [Tooltip("Show label as '60 FPS' if true, or just '60' if false.")]
    public bool showLabel = true;

    private Text fpsText;
    private float timer;
    private int frames;

    private void Awake()
    {
        SpawnUI();
    }

    private void SpawnUI()
    {
        // If we already have text (e.g. duplicated component), don't spawn again
        if (fpsText != null)
            return;

        // 1. Create Canvas that scales with screen size
        GameObject canvasGO = new GameObject("FPS Canvas");
        canvasGO.transform.SetParent(transform, false);
        Canvas canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 9999;

        CanvasScaler scaler = canvasGO.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = 0.5f;
        scaler.referencePixelsPerUnit = 100f;

        canvasGO.AddComponent<GraphicRaycaster>();

        // 2. Create Text object anchored to top-right
        GameObject textGO = new GameObject("FPS Text");
        textGO.transform.SetParent(canvasGO.transform, false);

        RectTransform rect = textGO.AddComponent<RectTransform>();
        // Anchor to top-right corner
        rect.anchorMin = new Vector2(1f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(1f, 1f);
        // Offset inward by padding (negative because pivot is top-right)
        rect.anchoredPosition = new Vector2(-padding.x, -padding.y);
        rect.sizeDelta = boxSize;

        Text txt = textGO.AddComponent<Text>();
        txt.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        txt.fontSize = fontSize;
        txt.alignment = TextAnchor.UpperRight;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;
        txt.verticalOverflow = VerticalWrapMode.Overflow;
        txt.color = textColor;
        txt.raycastTarget = false;
        txt.text = "0";

        // Add outline for readability on bright backgrounds
        Outline outline = textGO.AddComponent<Outline>();
        outline.effectColor = new Color(0f, 0f, 0f, 0.75f);
        outline.effectDistance = new Vector2(1f, 1f);

        fpsText = txt;
        ApplyAppearance();
    }

    private void OnValidate()
    {
        if (fpsText != null)
            ApplyAppearance();
    }

    private void ApplyAppearance()
    {
        fpsText.fontSize = fontSize;
        fpsText.color = textColor;
        RectTransform rect = fpsText.rectTransform;
        rect.anchoredPosition = new Vector2(-padding.x, -padding.y);
        rect.sizeDelta = boxSize;
    }

    private void Update()
    {
        if (fpsText == null)
            return;

        frames++;
        timer += Time.unscaledDeltaTime;

        if (timer >= updateInterval)
        {
            // floored to int as requested
            float avgFps = frames / timer;
            int fps = Mathf.FloorToInt(avgFps);

            fpsText.text = showLabel ? fps + " FPS" : fps.ToString();

            frames = 0;
            timer = 0f;
        }
    }
}
