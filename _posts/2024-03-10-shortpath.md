---
layout: post
title: 백준 4212 최단 비행 경로
date: 2024-03-10 14:05:00 +0900
categories:
- PS
tags:
- PS
description: 기하학, 스위핑, 데이크스트라
usemathjax: true
---

# BOJ 4212 최단 비행 경로

{% include rate.html image_path="/assets/images/rate/R5.svg" url="https://www.acmicpc.net/problem/4212" discription="4212 최단 비행 경로"%}

사용 알고리즘 :
- Geometry
- sweeping
- dijkstra

3차원 기하학과 데이크스트라가 적절히 섞여있는 문제입니다. 구사과 님의 코드를 해석하고 공부해서 구면좌표계를 다루는 방법을 터득하고, 예전에 풀었던 원반 위의 데이크스트라 문제를 응용해서 원반 위에서의 스위핑과 데이크스트라를 구현했습니다.

## 기하학

구조체 `Pos3D`를 만들어 3차원 기하학을 구현합니다. 연산자를 오버로딩해서 쓰면 함수들이 간결해지므로 연산자들을 오버로딩하겠습니다. 단위 벡터, 벡터의 크기를 구하는 함수도 만들어줍니다. 그리고 구면좌표계와 직교좌표계 사이의 변환 함수를 만들어줍니다.

{% highlight cpp %}
//C++
struct Pos3D {
    ld x, y, z;
    Pos3D(ld X = 0, ld Y = 0, ld Z = 0) : x(X), y(Y), z(Z) {}
    //bool operator == (const Pos3D& p) const { return zero(x - p.x) && zero(y - p.y) && zero(z - p.z); }
    //bool operator != (const Pos3D& p) const { return !zero(x - p.x) || !zero(y - p.y) || !zero(z - p.z); }
    //bool operator < (const Pos3D& p) const { return zero(x - p.x) ? zero(y - p.y) ? z < p.z : y < p.y : x < p.x; }
    ld operator * (const Pos3D& p) const { return x * p.x + y * p.y + z * p.z; }//dot product
    Pos3D operator / (const Pos3D& p) const {//cross product
        Pos3D ret;
        ret.x = y * p.z - z * p.y;
        ret.y = z * p.x - x * p.z;
        ret.z = x * p.y - y * p.x;
        return ret;
    }
    Pos3D operator + (const Pos3D& p) const { return { x + p.x, y + p.y, z + p.z }; }
    Pos3D operator - (const Pos3D& p) const { return { x - p.x, y - p.y, z - p.z }; }
    Pos3D operator * (const ld& scalar) const { return { x * scalar, y * scalar, z * scalar }; }
    Pos3D operator / (const ld& scalar) const { return { x / scalar, y / scalar, z / scalar }; }
    //Pos3D& operator += (const Pos3D& p) { x += p.x; y += p.y; z += p.z; return *this; }
    //Pos3D& operator *= (const ld& scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    ld Euc() const { return x * x + y * y + z * z; }
    ld mag() const { return sqrtl(Euc()); }
    Pos3D unit() const { return *this / mag(); }
};
const Pos3D O = { 0, 0, 0 };
const Pos3D MAXP3D = { INF, INF, INF };
std::vector<Pos3D> pos;
Pos3D S2C(const ld& lon, const ld& lat) {//Spherical to Cartesian
    ld phi = lon * PI / 180;
    ld the = lat * PI / 180;
    return Pos3D(cos(phi) * cos(the), sin(phi) * cos(the), sin(the));
}
{% endhighlight %}

외적과 내적, 벡터 더하기, 빼기 등도 구현했습니다.

## 스위핑

구면 위에 있는 원 위로만 움직일 수 있으므로, 원의 중심과 중심 사이는 두 원이 이어져있을 때 건너갈 수 있습니다. 만약 두 원이 떨어져있고, 사이에 다른 원들이 있는 경우라면? 사이에 있는 원들 위로 가면서 모든 원이 이어져있다면 출발한 원에서 목표 원까지 도착할 수 있게 됩니다. <br/>일단 두 윈의 교점을 찾아보겠습니다.

![spath1](/assets/images/2024-03-10-sphere-dijk/spath01.jpg)

두 원의 중심과 구의 중심 사이의 거리는 1이라고 하겠습니다. 이제부터 설명하는 모든 벡터는 크기를 언급하지 않는 이상 원점 기준 1의 거리로 떨어져있다고 생각하면 됩니다. <br/>두 원의 중심의 중점을 구해줍니다. 중점과 구의 원점이 같은 경우는 두 원이 서로 반대 방향에 있는 상황이므로 교점이 없거나 무한하므로 교점을 구하지 않고 함수를 종료합니다. <br/>두 원의 중심의 중심을 `mid`라고 하겠습니다. `mid`의 크기가 의미하는 것은 두 원 중 하나의 중심으로부터 `mid`까지 이동했을 때의 호의 각도의 `cos`값이 됩니다.

![spath2](/assets/images/2024-03-10-sphere-dijk/spath02.jpg)

비행기의 허용 이동 거리를 호로 표현했을 때 호의 각도 \\(theta\\)의 `cos`값과 비교해줍니다. -1 에서 1 사이에 있지 않다면 허용 거리를 초과해서 이동해야한다는 뜻이 되므로 교점이 존재하지 않습니다. 나머지는 피타고라스의 공식을 응용해 두 교점을 구합니다. 교점이 하나만 발생하는 경우도 있을 수 있어 조금 특수하게 구했습니다.<br/>말로만 설명하긴 어려울 것 같아 그림을 그렸습니다만 그래도 이해가 어렵다면 직접 그려보시길 바랍니다.
<br/>
스위핑을 하기 위해서는 두 원을 이은 경로 사이에 있는 모든 교점을 구해줄 필요가 있습니다. 두 원의 외적을 구해서 평면의 법선 벡터를 구해줍니다. 법선 벡터를 목표 원의 중심을 표현하는 벡터에 사영하고, 그 크기를 법선 벡터에 적용해서 빼주면 목표 원으로부터 평명까지의 수선의 발을 내릴 수 있게 됩니다. 수선의 발의 크기는 위에서 설명한 호의 각도의 `cos`값과 같은 맥락이 됩니다. 이후는 위와 같은 과정으로 교점을 구해줍시다.

![spath3](/assets/images/2024-03-10-sphere-dijk/spath03.jpg)

![spath4](/assets/images/2024-03-10-sphere-dijk/spath04.jpg)

{% highlight cpp %}
bool circle_intersection(const Pos3D& a, const Pos3D& b, const ld& th, std::vector<Pos3D>& inxs) {
    inxs.clear();
    Pos3D mid = (a + b) * .5;
    if (zero(mid.mag())) return 0;
    ld x = cos(th) / mid.mag();
    if (x < -1 || 1 < x) return 0;
    Pos3D w = mid.unit() * x;
    ld ratio = sqrtl(1 - x * x);
    Pos3D h = (mid / (b - a)).unit() * ratio;
    inxs.push_back(w + h);
    if (!zero(ratio)) inxs.push_back(w - h);
    return 1;
}
bool plane_circle_intersection(const Pos3D& a, const Pos3D& perp, const ld& th, std::vector<Pos3D>& inxs) {
    inxs.clear();
    Pos3D vec = a - (perp * (perp * a));
    if (zero(vec.mag())) return 0;
    ld x = cos(th) / vec.mag();
    if (x < -1 || 1 < x) return 0;
    Pos3D w = vec.unit() * x;
    ld ratio = sqrtl(1 - x * x);
    Pos3D h = (vec.unit() / perp) * ratio;
    inxs.push_back(w + h);
    if (!zero(ratio)) inxs.push_back(w - h);
    return 1;
}
{% endhighlight %}

구면에서 두 원의 교점, 한 원과 한 평면의 교점까지 찾을 수 있게 되었습니다. <br/>교점들을 구했다면 이제 교점들을 스위핑해서 두 원의 중심을 잇는 경로가 이어질 수 있는지 판단해줍니다.

![spath5](/assets/images/2024-03-10-sphere-dijk/spath05.jpg)

지구본을 펴서 세계지도를 그리듯이 전개해봅니다. 가장 왼쪽 원에서 가장 오른쪽 원까지의 최단 경로는 위와 같이 그려집니다. 경로는 두 개의 세부 경로로 이루어져있습니다. 이 중 왼쪽 경로에 대한 스위핑을 해보겠습니다.

![spath6](/assets/images/2024-03-10-sphere-dijk/spath06.jpg)

이 경로가 존재하는 평면에서 생기는 모든 교점을 찾아줍니다. 시작점의 각도를 0, 도착점의 각도는 ang, 그리고 다른 모든 교점들의 각도는 두 점이 만든 각도 기준으로 계산해줍니다. <br/>원의 시작에서 생기는 교점에는 -1, 끝에서 생기는 교점에는 1을 부여하고, 시작점과 도착점에는 0을 부여해서 컨테이너에 넣고 정렬해줍니다. 하나씩 하나씩 꺼내면서 시작점이 원 안에 들어있는지, 도중에 모든 원을 벗어나는지, 도착점까지 도착할 수 있는지를 보고, 가능하다면 1, 불가능하다면 0을 반환합니다.

![spath7](/assets/images/2024-03-10-sphere-dijk/spath07.jpg)

{% highlight cpp %}
Pos3D point(const Pos3D Xaxis, const Pos3D Yaxis, const ld& th) {
    return Xaxis * cos(th) + Yaxis * sin(th);
}
ld angle(const Pos3D Xaxis, const Pos3D Yaxis, const Pos3D& p) {
    ld X = Xaxis * p;
    ld Y = Yaxis * p;
    ld th = atan2(Y, X);
    return th;
}
Pos3D cross(const Pos3D& d1, const Pos3D& d2, const Pos3D& d3) { return (d2 - d1) / (d3 - d2); }
int ccw(const Pos3D& d1, const Pos3D& d2, const Pos3D& d3, const Pos3D& norm) {
    Pos3D CCW = cross(d1, d2, d3);
    ld ret = CCW * norm;
    return zero(ret) ? 0 : ret > 0 ? 1 : -1;
}
bool inner_check(const Pos3D& d1, const Pos3D& d2, const Pos3D& t, const Pos3D& nrm) {
    return ccw(O, d1, t, nrm) >= 0 && ccw(O, d2, t, nrm) <= 0;
}
bool connectable(const std::vector<Pos3D>& P, const Pos3D& a, const Pos3D& b, const ld& th) {
    if (zero((a - b).mag())) return 1;
    if (zero((a + b).mag()) && R > ERAD * PI * .5 - TOL) return 1;
    Pos3D perp = (a / b).unit();
    Pos3D X = a.unit();//X-axis
    Pos3D Y = (perp / a).unit();//Y-axis
    ld ang = angle(X, Y, b);
    std::vector<Info> tmp = { { 0, 0 }, { 0, ang } };
    std::vector<Pos3D> inxs;
    for (int i = 0; i < N; i++) {//sweeping
        if (acos(a * P[i]) < th + TOL && acos(b * P[i]) < th + TOL) return 1;
        if (plane_circle_intersection(P[i], perp, th, inxs)) {
            if (inxs.size() == 1) continue;
            Pos3D axis = (P[i] / perp).unit();
            Pos3D mid = (perp / axis).unit();
            Pos3D hi = inxs[0], lo = inxs[1];
            if (ccw(O, mid, lo, perp) > 0) std::swap(hi, lo);
            ld h = angle(X, Y, hi), l = angle(X, Y, lo);
            if (inner_check(a, b, lo, perp) &&
                inner_check(a, b, hi, perp)) {
                if (h < l) {
                    tmp.push_back({ 1, -INF });
                    tmp.push_back({ -1, l });
                    tmp.push_back({ 1, h });
                    tmp.push_back({ -1, INF });
                }
                else {
                    tmp.push_back({ 1, l });
                    tmp.push_back({ -1, h });
                }
            }
            else if (inner_check(a, b, lo, perp)) {
                if (h < 0) h += 2 * PI;
                tmp.push_back({ 1, l });
                tmp.push_back({ -1, h });
            }
            else if (inner_check(a, b, hi, perp)) {
                if (l > 0) l -= 2 * PI;
                tmp.push_back({ 1, l });
                tmp.push_back({ -1, h });
            }
        }
    }
    std::sort(tmp.begin(), tmp.end());
    int toggle = 0;
    bool f = 0;
    for (const Info& s : tmp) {
        toggle -= s.i;
        if (!s.i) f = !f;
        if (f && toggle <= 0) return 0;
    }
    return 1;
}
{% endhighlight %}

모든 경로를 수학 계산과 스위핑으로 구해줍니다.

{% highlight cpp %}
void solve(const int& tc) {
    std::cout << "Case " << tc << ":\n";
    TH = R / ERAD;
    pos.resize(N);
    ld lon, lat;
    for (int i = 0; i < N; i++)
        std::cin >> lon >> lat, pos[i] = S2C(lon, lat);//unit

    std::vector<Pos3D> inxs;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (circle_intersection(pos[i], pos[j], TH, inxs))
                for (const Pos3D& inx : inxs) pos.push_back(inx);
    int sz = pos.size();
    for (int i = 0; i < sz; i++) {
        adj[i][i] = 0;
        for (int j = 0; j < i; j++) {
            if (connectable(pos, pos[i], pos[j], TH))
                adj[i][j] = adj[j][i] = ERAD * acos(pos[i] * pos[j]);
            else
                adj[i][j] = adj[j][i] = INF;
        }
    }
}
{% endhighlight %}

이제 지도를 완성한 후 쿼리를 받고 두 점 사이의 최단 경로를 구하는 일만 남았습니다.

## 데이크스트라

두 공항 사이의 최단 경로를 구하기 위해서는 먼저 모든 점들에 대해 최단 경로를 구해줘야 합니다. 그 전에 생각해보면, 원과 원에 의해 생기는 교점들은 최종 쿼리를 수행할 때는 고려할 필요가 없습니다. 허용 거리 안에서 도달할 수 있는 '공항'만 연료를 충전할 수 있기 때문입니다.<br/> 그러니 이렇게 해보겠습니다.

1. 일단 N개의 공항에 대해 다른 모든 점으로 가는 최단 경로를 데이크스트라를 시행해서 구한다.
2. 데이크스트라 결과로 구한 비용 중 0 ~ N - 1 번 비용은 출발 공항에서 각 도착 공항들에 대한 비용 그래프이다.
3. 이를 각 출발점에 대해 N번 구하고 각각 복사해 최종본을 만든다.
4. 구해낸 최종 지도와 한계 허용 거리를 가지고 쿼리를 수행한다.

이렇게 해주면 복잡한 구현 없이도 공항들만 거쳐가는 최단 경로들을 모두 찾을 수 있습니다.

{% highlight cpp %}
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
//#include <cstring>
typedef long long ll;
//typedef long double ld;
typedef double ld;
const ld INF = 1e17;
const ld TOL = 1e-7;
const ld PI = acos(-1);
const ld ERAD = 6370;
const int LEN = 600 + 25;//nC2(25) + 25
bool zero(const ld& x) { return std::abs(x) < TOL; }
ld norm(ld& th) {
    while (th < -TOL) th += PI * 2;
    while (th > PI * 2) th -= PI * 2;
    return th;
}

int N, T, q;
ld R, TH;
ld adj[LEN][LEN], COST[LEN], G[25][25];
struct Info {
    int i;
    ld c;
    Info(int I = 0, ld C = 0) : i(I), c(C) {}
    bool operator < (const Info& x) const { return zero(c - x.c) ? i < x.i : c > x.c; }
};
std::priority_queue<Info> Q;
void dijkstra_adj(const int& v, const int& sz, const int& n = N) {
    for (int i = 0; i < sz; i++) COST[i] = INF;
    Q.push({ v, 0 });
    COST[v] = 0;
    while (Q.size()) {
        Info p = Q.top(); Q.pop();
        if (p.c > COST[p.i]) continue;
        for (int i = 0; i < sz; i++) {
            ld w = adj[p.i][i];
            if (w > INF - 1) continue;
            ld cost = p.c + w;
            if (COST[i] > cost) {
                COST[i] = cost;
                Q.push({ i, cost });
            }
        }
    }
    for (int g = 0; g < n; g++) G[v][g] = COST[g];
    return;
}
ld dijkstra(const int& v, const int& g, const int& sz, const ld& limit) {
    for (int i = 0; i < sz; i++) COST[i] = INF;
    Q.push({ v, 0 });
    COST[v] = 0;
    while (Q.size()) {
        Info p = Q.top(); Q.pop();
        if (p.c > COST[p.i]) continue;
        for (int i = 0; i < sz; i++) {
            ld w = G[p.i][i];
            if (w > limit) continue;
            ld cost = p.c + w;
            if (COST[i] > cost) {
                COST[i] = cost;
                Q.push({ i, cost });
            }
        }
    }
    return COST[g];
}

...

void query() {
    int s, t; ld c;
    std::cin >> s >> t >> c;
    ld ans = dijkstra(s - 1, t - 1, N, c);
    if (ans > 1e16) std::cout << "impossible\n";
    else std::cout << ans << "\n";
    return;
}

...
void solve(const int& tc) {
    std::cout << "Case " << tc << ":\n";
    
    ...

    for (int i = 0; i <br N; i++) dijkstra_adj(i, sz);

    std::cin >> q;
    while (q--) query();
    return;
}
{% endhighlight %}

3차원 기하학과 데이크스트라, 정점 분리 등의 개념을 이해하면 재밌게 풀 수 있는 문제입니다.<br/>
풀이를 온전히 제가 생각해낸 건 아니고, 구사과님의 코드를 참고해서 구현 방법을 연구했고 별도의 기하학 공부를 병행해서 풀었습니다. <br/>스위핑과 데이크스트라를 직접 구현해서 참고했던 원본 코드의 약 11배의 성능 개선을 이뤄냈으니 만족합니다.

![spath8](/assets/images/2024-03-10-sphere-dijk/goood.png)

전체 코드

{% highlight cpp %}
//C++
#include <iostream>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
//#include <cstring>
typedef long long ll;
//typedef long double ld;
typedef double ld;
const ld INF = 1e17;
const ld TOL = 1e-7;
const ld PI = acos(-1);
const ld ERAD = 6370;
const int LEN = 600 + 25;//nC2(25) + 25
bool zero(const ld& x) { return std::abs(x) < TOL; }
ld norm(ld& th) {
    while (th < -TOL) th += PI * 2;
    while (th > PI * 2) th -= PI * 2;
    return th;
}

int N, T, q;
ld R, TH;
ld adj[LEN][LEN], COST[LEN], G[25][25];
struct Info {
    int i;
    ld c;
    Info(int I = 0, ld C = 0) : i(I), c(C) {}
    bool operator < (const Info& x) const { return zero(c - x.c) ? i < x.i : c > x.c; }
};
std::priority_queue<Info> Q;
void dijkstra_adj(const int& v, const int& sz, const int& n = N) {
    for (int i = 0; i < sz; i++) COST[i] = INF;
    Q.push({ v, 0 });
    COST[v] = 0;
    while (Q.size()) {
        Info p = Q.top(); Q.pop();
        if (p.c > COST[p.i]) continue;
        for (int i = 0; i < sz; i++) {
            ld w = adj[p.i][i];
            if (w > INF - 1) continue;
            ld cost = p.c + w;
            if (COST[i] > cost) {
                COST[i] = cost;
                Q.push({ i, cost });
            }
        }
    }
    for (int g = 0; g < n; g++) G[v][g] = COST[g];
    return;
}
ld dijkstra(const int& v, const int& g, const int& sz, const ld& limit) {
    for (int i = 0; i < sz; i++) COST[i] = INF;
    Q.push({ v, 0 });
    COST[v] = 0;
    while (Q.size()) {
        Info p = Q.top(); Q.pop();
        if (p.c > COST[p.i]) continue;
        for (int i = 0; i < sz; i++) {
            ld w = G[p.i][i];
            if (w > limit) continue;
            ld cost = p.c + w;
            if (COST[i] > cost) {
                COST[i] = cost;
                Q.push({ i, cost });
            }
        }
    }
    return COST[g];
}
struct Pos3D {
    ld x, y, z;
    Pos3D(ld X = 0, ld Y = 0, ld Z = 0) : x(X), y(Y), z(Z) {}
    //bool operator == (const Pos3D& p) const { return zero(x - p.x) && zero(y - p.y) && zero(z - p.z); }
    //bool operator != (const Pos3D& p) const { return !zero(x - p.x) || !zero(y - p.y) || !zero(z - p.z); }
    //bool operator < (const Pos3D& p) const { return zero(x - p.x) ? zero(y - p.y) ? z < p.z : y < p.y : x < p.x; }
    ld operator * (const Pos3D& p) const { return x * p.x + y * p.y + z * p.z; }
    Pos3D operator / (const Pos3D& p) const {
        Pos3D ret;
        ret.x = y * p.z - z * p.y;
        ret.y = z * p.x - x * p.z;
        ret.z = x * p.y - y * p.x;
        return ret;
    }
    Pos3D operator + (const Pos3D& p) const { return { x + p.x, y + p.y, z + p.z }; }
    Pos3D operator - (const Pos3D& p) const { return { x - p.x, y - p.y, z - p.z }; }
    Pos3D operator * (const ld& scalar) const { return { x * scalar, y * scalar, z * scalar }; }
    Pos3D operator / (const ld& scalar) const { return { x / scalar, y / scalar, z / scalar }; }
    //Pos3D& operator += (const Pos3D& p) { x += p.x; y += p.y; z += p.z; return *this; }
    //Pos3D& operator *= (const ld& scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    ld Euc() const { return x * x + y * y + z * z; }
    ld mag() const { return sqrtl(Euc()); }
    Pos3D unit() const { return *this / mag(); }
};
const Pos3D O = { 0, 0, 0 };
const Pos3D MAXP3D = { INF, INF, INF };
std::vector<Pos3D> pos;
Pos3D S2C(const ld& lon, const ld& lat) {//Spherical to Cartesian
    ld phi = lon * PI / 180;
    ld the = lat * PI / 180;
    return Pos3D(cos(phi) * cos(the), sin(phi) * cos(the), sin(the));
}
bool circle_intersection(const Pos3D& a, const Pos3D& b, const ld& th, std::vector<Pos3D>& inxs) {
    inxs.clear();
    Pos3D mid = (a + b) * .5;
    if (zero(mid.mag())) return 0;
    ld x = cos(th) / mid.mag();
    if (x < -1 || 1 < x) return 0;
    Pos3D w = mid.unit() * x;
    ld ratio = sqrtl(1 - x * x);
    Pos3D h = (mid / (b - a)).unit() * ratio;
    inxs.push_back(w + h);
    if (!zero(ratio)) inxs.push_back(w - h);
    return 1;
}
bool plane_circle_intersection(const Pos3D& a, const Pos3D& perp, const ld& th, std::vector<Pos3D>& inxs) {
    inxs.clear();
    Pos3D vec = a - (perp * (perp * a));
    if (zero(vec.mag())) return 0;
    ld x = cos(th) / vec.mag();
    if (x < -1 || 1 < x) return 0;
    Pos3D w = vec.unit() * x;
    ld ratio = sqrtl(1 - x * x);
    Pos3D h = (vec.unit() / perp) * ratio;
    inxs.push_back(w + h);
    if (!zero(ratio)) inxs.push_back(w - h);
    return 1;
}
Pos3D point(const Pos3D Xaxis, const Pos3D Yaxis, const ld& th) {
    return Xaxis * cos(th) + Yaxis * sin(th);
}
ld angle(const Pos3D Xaxis, const Pos3D Yaxis, const Pos3D& p) {
    ld X = Xaxis * p;
    ld Y = Yaxis * p;
    ld th = atan2(Y, X);
    return th;
}
Pos3D cross(const Pos3D& d1, const Pos3D& d2, const Pos3D& d3) { return (d2 - d1) / (d3 - d2); }
int ccw(const Pos3D& d1, const Pos3D& d2, const Pos3D& d3, const Pos3D& norm) {
    Pos3D CCW = cross(d1, d2, d3);
    ld ret = CCW * norm;
    return zero(ret) ? 0 : ret > 0 ? 1 : -1;
}
bool inner_check(const Pos3D& d1, const Pos3D& d2, const Pos3D& t, const Pos3D& nrm) {
    return ccw(O, d1, t, nrm) >= 0 && ccw(O, d2, t, nrm) <= 0;
}
bool connectable(const std::vector<Pos3D>& P, const Pos3D& a, const Pos3D& b, const ld& th) {
    if (zero((a - b).mag())) return 1;
    if (zero((a + b).mag()) && R > ERAD * PI * .5 - TOL) return 1;
    Pos3D perp = (a / b).unit();
    Pos3D X = a.unit();//X-axis
    Pos3D Y = (perp / a).unit();//Y-axis
    ld ang = angle(X, Y, b);
    std::vector<Info> tmp = { { 0, 0 }, { 0, ang } };
    std::vector<Pos3D> inxs;
    for (int i = 0; i < N; i++) {//sweeping
        if (acos(a * P[i]) < th + TOL && acos(b * P[i]) < th + TOL) return 1;
        if (plane_circle_intersection(P[i], perp, th, inxs)) {
            if (inxs.size() == 1) continue;
            Pos3D axis = (P[i] / perp).unit();
            Pos3D mid = (perp / axis).unit();
            Pos3D hi = inxs[0], lo = inxs[1];
            if (ccw(O, mid, lo, perp) > 0) std::swap(hi, lo);
            ld h = angle(X, Y, hi), l = angle(X, Y, lo);
            if (inner_check(a, b, lo, perp) &&
                inner_check(a, b, hi, perp)) {
                if (h < l) {
                    tmp.push_back({ 1, -INF });
                    tmp.push_back({ -1, l });
                    tmp.push_back({ 1, h });
                    tmp.push_back({ -1, INF });
                }
                else {
                    tmp.push_back({ 1, l });
                    tmp.push_back({ -1, h });
                }
            }
            else if (inner_check(a, b, lo, perp)) {
                if (h < 0) h += 2 * PI;
                tmp.push_back({ 1, l });
                tmp.push_back({ -1, h });
            }
            else if (inner_check(a, b, hi, perp)) {
                if (l > 0) l -= 2 * PI;
                tmp.push_back({ 1, l });
                tmp.push_back({ -1, h });
            }
        }
    }
    std::sort(tmp.begin(), tmp.end());
    int toggle = 0;
    bool f = 0;
    for (const Info& s : tmp) {
        toggle -= s.i;
        if (!s.i) f = !f;
        if (f && toggle <= 0) return 0;
    }
    return 1;
}
void query() {
    int s, t; ld c;
    std::cin >> s >> t >> c;
    ld ans = dijkstra(s - 1, t - 1, N, c);
    if (ans > 1e16) std::cout << "impossible\n";
    else std::cout << ans << "\n";
    return;
}
void solve(const int& tc) {
    std::cout << "Case " << tc << ":\n";
    TH = R / ERAD;
    pos.resize(N);
    ld lon, lat;
    for (int i = 0; i < N; i++)
        std::cin >> lon >> lat, pos[i] = S2C(lon, lat);//unit

    std::vector<Pos3D> inxs;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            if (circle_intersection(pos[i], pos[j], TH, inxs))
                for (const Pos3D& inx : inxs) pos.push_back(inx);
    int sz = pos.size();
    for (int i = 0; i < sz; i++) {
        adj[i][i] = 0;
        for (int j = 0; j < i; j++) {
            if (connectable(pos, pos[i], pos[j], TH))
                adj[i][j] = adj[j][i] = ERAD * acos(pos[i] * pos[j]);
            else
                adj[i][j] = adj[j][i] = INF;
        }
    }

    for (int i = 0; i < N; i++) dijkstra_adj(i, sz);

    std::cin >> q;
    while (q--) query();
    return;
}
void solve() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cout << std::fixed;
    std::cout.precision(3);
    T = 0;
    while (std::cin >> N >> R) solve(++T);
    return;
}
int main() { solve(); return 0; }//boj4212 Shortest Flight Path
{% endhighlight %}
