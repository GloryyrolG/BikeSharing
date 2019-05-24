# Glory Chen
# May, 23th, 2019
# Version-0

import numpy as np

P = 8
TK = 4
n = 55
T = 0.02
DT = 0.25
L = 10
K = 25
C = 0.5
c1 = 10000
c2 = 100
c3 = 1

M = 50
G = 100
p_m = 0.01

J = 100
at = 0.999
e = 0.1 ** 30

def Pave(Dist, curr, indx, nxt, pos):
    assert indx <= P + 1
    I = np.zeros([P + 2, ], dtype = int)
    Id = np.zeros([P + 2, ], dtype = int)
    ia = np.zeros([P + 2, ], dtype = int)
    k = indx
    x = pos
    while k <= P and x * DT <= T * Dist[curr][nxt]:
        I[k], Id[k], ia[k] = curr, 0 if x > 1 else 1, 0
        k += 1
        x += 1
    I[k] = nxt
    if k <= P:
        Id[k], ia[k] = 0 if x > 1 else 1, 1
    k += 1
    return k - indx, I, Id, ia

def Fill(Dist, curr, indx):
    assert indx <= P + 1
    I = np.zeros([P + 2, ], dtype = int)
    Id = np.zeros([P + 2, ], dtype = int)
    ia = np.zeros([P + 2, ], dtype = int)
    k = indx
    while k <= P:
        nxt = np.random.randint(1, n + 1)
        Len, tI, tId, tia = Pave(Dist, curr, k, nxt, 1)
        I[k: k + Len], Id[k: k + Len], ia[k: k + Len] = tI[k: k + Len], tId[k: k + Len], tia[k: k + Len]
        k += Len
        curr = nxt
    if k == P + 1:
        I[-1] = I[-2]
    return I, Id, ia

def DAInit(Dist, init):
    I = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
    Id = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
    ia = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
    for i in range(1, M + 1):
        for j in range(1, TK + 1):
            I[i][j][0] = init[j][0]
            I[i][j], Id[i][j], ia[i][j] = Fill(Dist, I[i][j][0], 1)
    return I, Id, ia

def GetRw(b0, cf, I, ia, ld):
    rw = 0
    pen = 0
    ed = np.array(b0)
    for i in range(1, P + 1):
        ed += cf[: , i]
        al = np.where(ia[: , i])
        for j in al[0]:
            ed[I[j][i]] -= ld[j][i]
        rw += sum(ed[ed > K] - K) + sum(-ed[ed < 0]) * 2
        ed[ed > K], ed[ed < 0] = K, 0
    return rw, pen

def GetX(Dist, init, b0, cf, I, ia, ld):
    ld1 = np.array(ld)
    sh = np.array(init[: , 1])
    ed = np.array(b0)
    for i in range(1, P + 1):
        ed += cf[: , i]
        for j in range(1, TK + 1):
            if ia[j][i] == 1:
                s = I[j][i]
                low = max([-sh[j], min([ed[s] - K, 0])]) - ld1[j][i]
                up = min([L - sh[j], max([ed[s], 0])]) - ld1[j][i]
                dlt = np.random.randint(-up, low + 1)
                ld1[j][i] += dlt
                sh[j] += ld1[j][i]
                ed[s] -= ld1[j][i]
        ed[ed > K], ed[ed < 0] = K, 0
    return ld1

def SA(Dist, init, b0, cf, I, Id, ia):
    ld = np.zeros([TK + 1, P + 1], dtype = int)
    rw, pen = GetRw(b0, cf, I, ia, ld)
    Tsa = 1
    for j in range(1, J + 1):
        ld1 = GetX(Dist, init, b0, cf, I, ia, ld)
        rw1, pen1 = GetRw(b0, cf, I, ia, ld1)
        df = c1 * rw1 + c3 * pen1 - (c1 * rw + c3 * pen)
        if df < 0 or np.random.rand(1) <= np.exp(-df / Tsa):
            ld, rw, pen = ld1, rw1, pen1
        Tsa *= at
        if Tsa < e:
            break
    cost = 0
    for i in range(1, TK + 1):
        for k in range(1, P):
            cost += C * Dist[I[i][k]][I[i][k + 1]]
        if ia[i][P] == 0:
            for k in range(P, 0, -1):
                if Id[i][k] == 1:
                    cost += C * (P - k + 1) / T
                    break
    return c1 * rw + c2 * cost + c3 * pen

def Concat(Dist, curr, indx, nxt, st, I, Id, ia):
    assert indx <= P + 1
    I1, Id1, ia1 = np.array(I[0]), np.array(Id[0]), np.array(ia[0])
    k = indx
    Len, tI, tId, tia = Pave(Dist, curr, k, nxt, 1)
    I1[k: k + Len], Id1[k: k + Len], ia1[k: k + Len] = tI[k: k + Len], tId[k: k + Len], tia[k: k + Len]
    k += Len
    if st <= P + 1:
        Len = min([P - k + 1, P - st + 1]) 
        I1[k: k + Len], Id1[k: k + Len], ia1[k: k + Len] = I[1][st: st + Len], Id[1][st: st + Len], ia[1][st: st + Len]
        k += Len
        if k <= P and ia1[k - 1] == 0:
            for j in range(k - 1, 0, -1):
                if Id1[j] == 1:
                    pos = k - j + 1
                    break
            Len, tI, tId, tia = Pave(Dist, I1[k - 1], k, I[1][-1], pos)
            I1[k: k + Len], Id1[k: k + Len], ia1[k: k + Len] = tI[k: k + Len], tId[k: k + Len], = tia[k: k + Len]
            k += Len
    if k == P + 1:
        if ia1[k - 1] == 1:
            I1[k] = I1[k - 1]
        else:
            I1[k] = I[1][st + Len]
        k += 1
    return k - indx, I1, Id1, ia1

def Cross(Dist, I, Id, ia):
    s = np.random.randint(1, TK + 1, size = 2)
    p = np.random.randint(1, P + 1)
    I1, Id1, ia1 = np.array(I), np.array(Id), np.array(ia)
    for i in range(2):
        for j in range(p, P + 2):
            if j == P + 1 or ia[1 - i][s[1 - i]][j] == 1:
                nxt = I[1 - i][s[1 - i]][j]
                st = j + 1
                break
        indx = -1
        for j in range(p, 0, -1):
            if Id[i][s[i]][j] == 1:
                curr = I[i][s[i]][j - 1]
                indx = j
                break
        assert indx != -1
        Len, tI, tId, tia = Concat(Dist, curr, indx, nxt, st,\
            I[[i, 1 - i], [s[i], s[1 - i]]], Id[[i, 1 - i], [s[i], s[1 - i]]], ia[[i, 1 - i], [s[i], s[1 - i]]])
        I1[i][s[i]][indx: indx + Len], Id1[i][s[i]][indx: indx + Len], ia1[i][s[i]][indx: indx + Len]\
            = tI[indx: indx + Len], tId[indx: indx + Len], tia[indx: indx + Len]
        indx += Len
        assert indx != P + 1
        if indx < P + 1:
            assert ia1[i][s[i]][indx - 1] == 1
            tI, tId, tia = Fill(Dist, I1[i][s[i]][indx - 1], indx)
            I1[i][s[i]][indx: ], Id1[i][s[i]][indx: ], ia1[i][s[i]][indx: ] = tI[indx: ], tId[indx: ], tia[indx: ]
    return I1, Id1, ia1

def Variation(Dist, I, Id, ia):
    I1, Id1, ia1 = np.array(I), np.array(Id), np.array(ia)
    s = np.random.randint(1, TK + 1)
    p = np.random.randint(1, P + 1)
    curr = np.random.randint(1, n + 1)
    indx = 1
    Len, tI, tId, tia = Pave(Dist, I[s][0], indx, curr, 1)
    I1[s][indx: indx + Len], Id1[s][indx: indx + Len], ia1[s][indx: indx + Len]\
        = tI[indx: indx + Len], tId[indx: indx + Len], tia[indx: indx + Len]
    indx += Len
    if indx > P + 1:
        return I1, Id1, ia1
    for i in range(p, P + 2):
        if i == P + 1 or ia[s][i] == 1:
            nxt = I[s][i]
            st = i + 1
            break
    Len, tI, tId, tia = Concat(Dist, curr, indx, nxt, st,\
        np.array([I1[s], I[s]]), np.array([Id1[s], Id[s]]), np.array([ia1[s], ia[s]]))
    I1[s][indx: indx + Len], Id1[s][indx: indx + Len], ia1[s][indx: indx + Len]\
        = tI[indx: indx + Len], tId[indx: indx + Len], tia[indx: indx + Len]
    indx += Len
    if indx < P + 1:
        for i in range(1, P + 2):
            if i == P + 1 or ia[s][i] == 1:
                nxt = I[s][i]
                st = i + 1
                break
        Len, tI, tId, tia = Concat(Dist, I1[s][indx - 1], indx, nxt, st,\
            np.array([Id1[s], I[s]]), np.array([Id1[s], Id[s]]), np.array([ia1[s], ia[s]]))
        I1[s][indx: indx + Len], Id1[s][indx: indx + Len], ia1[s][indx: indx + Len]\
            = tI[indx: indx + Len], tId[indx: indx + Len], tia[indx: indx + Len]
        indx += Len
        if indx < P + 1:
            tI, tId, tia = Fill(Dist, I1[s][indx - 1], indx)
            I1[s][indx: ], Id1[s][indx: ], ia1[s][indx: ] = tI[indx: ], tId[indx: ], tia[indx: ]
    return I1, Id1, ia1

def GA(Dist, init, b, cf):
    I, Id, ia = DAInit(Dist, init)
    I0 = np.zeros([4 * M + 1, TK + 1, P + 2], dtype = int)
    Id0 = np.zeros([4 * M + 1, TK + 1, P + 2], dtype = int)
    ia0 = np.zeros([4 * M + 1, TK + 1, P + 2], dtype = int)
    for g in range(1, G + 1):
        print('G', g)
        fit = np.array([np.inf] + [SA(Dist, init, b0, cf, I[i], Id[i], ia[i]) for i in range(1, M + 1)])
        rk = np.argsort(-fit)
        I1 = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
        Id1 = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
        ia1 = np.zeros([M + 1, TK + 1, P + 2], dtype = int)
        for i in range(1, M + 1, 2):
            I1[i: i + 2], Id1[i: i + 2], ia1[i: i + 2] = Cross(Dist,\
                I[rk[i: i + 2]], Id[rk[i: i + 2]], ia[rk[i: i + 2]])
        I0[1: M + 1], I0[M + 1: 2 * M + 1] = I[1: ], I1[1: ]
        Id0[1: M + 1], Id0[M + 1: 2 * M + 1] = Id[1: ], Id1[1: ]
        ia0[1: M + 1], ia0[M + 1: 2 * M + 1] = ia[1: ], ia1[1: ]
        v = np.random.rand(2 * M + 1)
        v[0] = 1
        indx = 2 * M + 1
        for pos in np.where(v <= p_m)[0]:
            I0[indx], Id0[indx], ia0[indx] = Variation(Dist, I0[pos], Id0[pos], ia0[pos])
            indx += 1
        fit = np.array([np.inf] + [SA(Dist, init, b0, cf, I0[i], Id0[i], ia0[i]) for i in range(1, indx)])
        rk = np.argsort(-fit)
        I[1: ], Id[1: ], ia[1: ] = I0[rk[1: M + 1]], Id0[rk[1: M + 1]], ia0[rk[1: M + 1]]
    return I[1], Id[1], ia[1]

if __name__ == '__main__':
    Dist = 8 * np.random.rand(n + 1, n + 1)
    for i in range(1, n + 1):
        for j in range(1, i):
            Dist[i][j] = Dist[j][i]
        Dist[i][i] = 0
    init = np.array([[0, 0]] + [[1, 0]] * TK)
    b0 = [0] + [20] * n
    cf = np.random.randint(-5, 6, size = [n + 1, P + 1])
    cf[0] = np.zeros([P + 1, ], dtype = int)
    for i in range(1, len(cf)):
        cf[i].sort()
        if np.random.rand(1) <= 0.5:
            cf[i] *= -1
        cf[i][0] = 0
    ansI, ansId, ansia = GA(Dist, init, b0, cf)
    print(ansI)
    print(ansId)
    print(ansia)
