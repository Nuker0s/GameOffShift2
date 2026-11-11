using System.Collections.Generic;
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

    [Header("Outline")]
    public bool useOutline = true;
    public Color outlineColor = new Color(0f, 0f, 0f, 0.75f);
    [Tooltip("Outline thickness in pixels.")]
    public float outlineThickness = 1f;

    [Header("FPS")]
    [Tooltip("How many seconds back the AVG is calculated over. Old frame times outside this window are thrown away every frame.")]
    public float averageWindow = 5f;

    private Text fpsText;
    private Outline fpsOutline;

    // Sliding window: every frame we push the latest delta time and
    // throw away anything older than averageWindow. AVG = frames / total time.
    private readonly Queue<float> frameTimes = new Queue<float>();
    private float windowSum;

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
        txt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        txt.fontSize = fontSize;
        txt.alignment = TextAnchor.UpperRight;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;
        txt.verticalOverflow = VerticalWrapMode.Overflow;
        txt.color = textColor;
        txt.raycastTarget = false;
        txt.text = "0";

        // Add outline for readability on bright backgrounds
        Outline outline = textGO.AddComponent<Outline>();
        outline.effectColor = outlineColor;
        outline.effectDistance = new Vector2(outlineThickness, -outlineThickness);

        fpsText = txt;
        fpsOutline = outline;
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

        if (fpsOutline != null)
        {
            fpsOutline.enabled = useOutline;
            fpsOutline.effectColor = outlineColor;
            fpsOutline.effectDistance = new Vector2(outlineThickness, -outlineThickness);
        }
    }

    private void Update()
    {
        if (fpsText == null)
            return;

        // Current FPS = instant value from this frame (unscaled so timeScale doesn't lie)
        float dt = Time.unscaledDeltaTime;
        int current = dt > 0f ? Mathf.FloorToInt(1f / dt) : 0;

        // Sliding average: push new frame time, throw away anything older than the window.
        // Updates in realtime every frame — no measure-then-reset cycle.
        float window = Mathf.Max(averageWindow, 0.1f);
        frameTimes.Enqueue(dt);
        windowSum += dt;
        while (windowSum > window && frameTimes.Count > 0)
            windowSum -= frameTimes.Dequeue();

        int avg = windowSum > 0f
            ? Mathf.FloorToInt(frameTimes.Count / windowSum)
            : 0;

        fpsText.text = "FPS " + current + " | AVG " + avg;
    }
}
