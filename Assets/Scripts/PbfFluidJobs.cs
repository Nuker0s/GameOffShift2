using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;

// Burst jobs for PbfFluidLite. Requires the Burst package (com.unity.burst, which pulls in Mathematics).
//
// Layout notes
//  * Every array is indexed by particle index i in [0, n). Arrays may be LONGER than n (capacity), jobs only touch [0, n).
//  * Inputs are sanitised on the main thread (non-finite -> active = 0), so FloatMode.Fast (which may assume no NaN/Inf)
//    is safe here.
//  * Particles are bucketed into a STATIC grid (cols x rows, cell size = smoothing radius). Positions outside the grid are
//    clamped into the border cells. That can never produce a wrong neighbour (clamping never moves two particles further
//    apart, and every candidate is still distance-checked) - it only costs speed if many particles pile up outside.
//  * Items are sorted by cell index, so the 3 cells of one grid row are ONE contiguous range in cellItems:
//    a 3x3 neighbourhood is just 3 ranges.

public struct PbfTypeParams
{
    public float rho0;        // rest density for this fluid type
    public float stiffness;   // 0..1 relaxation on the position correction
    public float viscosity;   // 0..1 fraction of relative velocity removed per step (already scaled)
}

/// <summary>pred = x + (v + g*dt) * dt   (where the engine is about to take the particle if nothing else acts)</summary>
[BurstCompile(FloatMode = FloatMode.Fast, CompileSynchronously = true)]
public struct PbfPredictJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float2> x;
    [ReadOnly] public NativeArray<float2> v;
    [ReadOnly] public NativeArray<float> gs;       // effective gravity scale (body scale * type multiplier)
    public float2 gravity;
    public float dt;
    [WriteOnly] public NativeArray<float2> vPred;
    [WriteOnly] public NativeArray<float2> pos0;
    [WriteOnly] public NativeArray<float2> pred;

    public void Execute(int i)
    {
        float2 vP = v[i] + gravity * (gs[i] * dt);
        float2 p0 = x[i] + vP * dt;
        vPred[i] = vP;
        pos0[i] = p0;
        pred[i] = p0;
    }
}

/// <summary>Counting-sort of the particles into the static grid. Single-threaded: O(n + cells), a few microseconds.</summary>
[BurstCompile(CompileSynchronously = true)]
public struct PbfGridJob : IJob
{
    [ReadOnly] public NativeArray<float2> p;
    [ReadOnly] public NativeArray<byte> active;
    public NativeArray<int> cellOf;      // per particle, -1 = not in the grid
    public NativeArray<int> cellStart;   // cells + 1
    public NativeArray<int> cellFill;    // cells + 1 (scratch)
    public NativeArray<int> cellItems;   // per particle, sorted by cell
    public float2 origin;
    public float invCell;
    public int cols, rows, n;

    public void Execute()
    {
        int total = cols * rows;
        for (int c = 0; c <= total; c++) cellStart[c] = 0;

        float maxX = cols - 1, maxY = rows - 1;
        for (int i = 0; i < n; i++)
        {
            if (active[i] == 0) { cellOf[i] = -1; continue; }
            float2 q = (p[i] - origin) * invCell;
            int gx = (int)math.clamp(q.x, 0f, maxX);   // clamp: outside the grid -> border cell
            int gy = (int)math.clamp(q.y, 0f, maxY);
            int c = gy * cols + gx;
            cellOf[i] = c;
            cellStart[c + 1]++;
        }
        for (int c = 0; c < total; c++) cellStart[c + 1] += cellStart[c];
        for (int c = 0; c < total; c++) cellFill[c] = cellStart[c];
        for (int i = 0; i < n; i++)
        {
            int c = cellOf[i];
            if (c >= 0) cellItems[cellFill[c]++] = i;
        }
    }
}

/// <summary>Density + PBF lambda per particle.  lambda_i = -C_i / (sum |dC_i/dp_k|^2 + eps)</summary>
[BurstCompile(FloatMode = FloatMode.Fast, CompileSynchronously = true)]
public struct PbfDensityJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float2> pred;
    [ReadOnly] public NativeArray<float> mass;
    [ReadOnly] public NativeArray<int> type;
    [ReadOnly] public NativeArray<PbfTypeParams> types;
    [ReadOnly] public NativeArray<int> cellOf;
    [ReadOnly] public NativeArray<int> cellStart;
    [ReadOnly] public NativeArray<int> cellItems;
    public int cols, rows, typeCount;
    public float h, h2, poly6, spikyGrad, selfW, eps;
    public bool writeRatio;
    [WriteOnly] public NativeArray<float> lambda;
    [WriteOnly] public NativeArray<float> ratioOut;   // rho / rho0, only written when writeRatio

    public void Execute(int i)
    {
        int c0 = cellOf[i];
        if (c0 < 0) { lambda[i] = 0f; if (writeRatio) ratioOut[i] = 0f; return; }

        int cy0 = c0 / cols, cx0 = c0 - cy0 * cols;
        int x0 = math.max(cx0 - 1, 0), x1 = math.min(cx0 + 1, cols - 1);
        int y0 = math.max(cy0 - 1, 0), y1 = math.min(cy0 + 1, rows - 1);

        float2 pi = pred[i];
        float invRho0 = 1f / types[math.min(type[i], typeCount - 1)].rho0;
        float dens = mass[i] * selfW;
        float2 gI = new float2(0f, 0f);
        float s2 = 0f;

        for (int ny = y0; ny <= y1; ny++)
        {
            int rb = ny * cols;
            int kEnd = cellStart[rb + x1 + 1];
            for (int k = cellStart[rb + x0]; k < kEnd; k++)
            {
                int j = cellItems[k];
                if (j == i) continue;
                float2 d = pi - pred[j];
                float r2 = d.x * d.x + d.y * d.y;
                if (r2 >= h2 || r2 < 1e-12f) continue;
                float q = h2 - r2;
                float mj = mass[j];
                dens += mj * poly6 * q * q * q;
                float r = math.sqrt(r2);
                float hr = h - r;
                // dC_i/dp_i contribution of neighbour j = (m_j / rho0) * gradW
                float2 g = d * (spikyGrad * hr * hr / r * mj * invRho0);
                gI += g;
                s2 += g.x * g.x + g.y * g.y;
            }
        }

        float ratio = dens * invRho0;
        if (writeRatio) ratioOut[i] = ratio;
        float C = ratio - 1f;
        if (C <= 0f) { lambda[i] = 0f; return; }          // only resist compression
        s2 += gI.x * gI.x + gI.y * gI.y;
        lambda[i] = -C / (s2 + eps);
    }
}

/// <summary>dp_i = (1/rho0) * sum_j (m_j*lambda_i + m_i*lambda_j) * gradW_ij.  Double-buffered: reads pred, writes predNext.</summary>
[BurstCompile(FloatMode = FloatMode.Fast, CompileSynchronously = true)]
public struct PbfCorrectionJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float2> pred;
    [ReadOnly] public NativeArray<float> lambda;
    [ReadOnly] public NativeArray<float> mass;
    [ReadOnly] public NativeArray<int> type;
    [ReadOnly] public NativeArray<PbfTypeParams> types;
    [ReadOnly] public NativeArray<int> cellOf;
    [ReadOnly] public NativeArray<int> cellStart;
    [ReadOnly] public NativeArray<int> cellItems;
    public int cols, rows, typeCount;
    public float h, h2, spikyGrad, maxCorr;
    [WriteOnly] public NativeArray<float2> predNext;

    public void Execute(int i)
    {
        float2 pi = pred[i];
        int c0 = cellOf[i];
        if (c0 < 0) { predNext[i] = pi; return; }

        int cy0 = c0 / cols, cx0 = c0 - cy0 * cols;
        int x0 = math.max(cx0 - 1, 0), x1 = math.min(cx0 + 1, cols - 1);
        int y0 = math.max(cy0 - 1, 0), y1 = math.min(cy0 + 1, rows - 1);

        float li = lambda[i], mi = mass[i];
        float2 corr = new float2(0f, 0f);

        for (int ny = y0; ny <= y1; ny++)
        {
            int rb = ny * cols;
            int kEnd = cellStart[rb + x1 + 1];
            for (int k = cellStart[rb + x0]; k < kEnd; k++)
            {
                int j = cellItems[k];
                if (j == i) continue;
                float lj = lambda[j];
                if (li == 0f && lj == 0f) continue;
                float2 d = pi - pred[j];
                float r2 = d.x * d.x + d.y * d.y;
                if (r2 >= h2 || r2 < 1e-12f) continue;
                float r = math.sqrt(r2);
                float hr = h - r;
                corr += d * (spikyGrad * hr * hr / r * (mass[j] * li + mi * lj));
            }
        }

        PbfTypeParams tp = types[math.min(type[i], typeCount - 1)];
        corr *= tp.stiffness / tp.rho0;
        float mag = math.length(corr);
        if (mag > maxCorr) corr *= maxCorr / mag;
        predNext[i] = pi + corr;
    }
}

/// <summary>Velocity the constraint wants, relative to the engine's own step: vC = vPred + (pred - pos0)/dt.</summary>
[BurstCompile(FloatMode = FloatMode.Fast, CompileSynchronously = true)]
public struct PbfVelocityJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float2> vPred;
    [ReadOnly] public NativeArray<float2> pred;
    [ReadOnly] public NativeArray<float2> pos0;
    public float invDt, maxSpeed;
    [WriteOnly] public NativeArray<float2> vCon;

    public void Execute(int i)
    {
        float2 vP = vPred[i];
        float2 vC = vP + (pred[i] - pos0[i]) * invDt;
        float cap = math.max(maxSpeed, math.length(vP));   // limit how much we ADD, never brake an existing speed
        float sp = math.length(vC);
        if (sp > cap && sp > 1e-6f) vC *= cap / sp;
        vCon[i] = vC;
    }
}

/// <summary>XSPH viscosity, then F = m * (vOut - vPred) / dt  (the engine adds gravity itself).</summary>
[BurstCompile(FloatMode = FloatMode.Fast, CompileSynchronously = true)]
public struct PbfViscosityForceJob : IJobParallelFor
{
    [ReadOnly] public NativeArray<float2> vCon;
    [ReadOnly] public NativeArray<float2> vPred;
    [ReadOnly] public NativeArray<float2> pred;
    [ReadOnly] public NativeArray<float> mass;
    [ReadOnly] public NativeArray<int> type;
    [ReadOnly] public NativeArray<PbfTypeParams> types;
    [ReadOnly] public NativeArray<int> cellOf;
    [ReadOnly] public NativeArray<int> cellStart;
    [ReadOnly] public NativeArray<int> cellItems;
    public int cols, rows, typeCount;
    public float h2, invDt;
    [WriteOnly] public NativeArray<float2> force;

    public void Execute(int i)
    {
        int c0 = cellOf[i];
        if (c0 < 0) { force[i] = new float2(0f, 0f); return; }

        float2 vc = vCon[i];
        float visc = types[math.min(type[i], typeCount - 1)].viscosity;
        if (visc > 0f)
        {
            int cy0 = c0 / cols, cx0 = c0 - cy0 * cols;
            int x0 = math.max(cx0 - 1, 0), x1 = math.min(cx0 + 1, cols - 1);
            int y0 = math.max(cy0 - 1, 0), y1 = math.min(cy0 + 1, rows - 1);
            float2 pi = pred[i];
            float2 acc = new float2(0f, 0f);
            float wSum = 0f;
            for (int ny = y0; ny <= y1; ny++)
            {
                int rb = ny * cols;
                int kEnd = cellStart[rb + x1 + 1];
                for (int k = cellStart[rb + x0]; k < kEnd; k++)
                {
                    int j = cellItems[k];
                    if (j == i) continue;
                    float2 d = pi - pred[j];
                    float r2 = d.x * d.x + d.y * d.y;
                    if (r2 >= h2 || r2 < 1e-12f) continue;
                    float q = h2 - r2;
                    float w = q * q * q;
                    acc += vCon[j] * w;
                    wSum += w;
                }
            }
            if (wSum > 1e-12f) vc += (acc / wSum - vc) * visc;
        }

        force[i] = (vc - vPred[i]) * (mass[i] * invDt);
    }
}
