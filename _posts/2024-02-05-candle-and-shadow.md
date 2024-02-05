---
layout: post
title: 백준 18190 촛불과 그림자
date: 2024-02-05 11:59:00 +0900
categories:
- PS
tags:
- PS
description: BOJ 18190 촛불과 그림자
usemathjax: true
---

# BOJ 3527 Jungle Outpost

{% include rate.html image_path="/assets/images/rate/R5.svg" url="https://www.acmicpc.net/problem/18190" discription="18190 촛불과 그림자"%}

요즘 고난이도 문제를 푸느라 블로그 관리에 꽤 소홀했습니다. 미치지 않고서야 푸는 사람이 적은 문제라 볼 사람이 있겠나 싶지만 그래도 풀었으니 기록은 해야죠.
[백준 27957 성벽 쌓기](https://www.acmicpc.net/problem/27957)와 함께 기하학 카테고리에서 풀고 싶었던 두 개의 문제 중 하나입니다. 성벽 쌓기는 N이 작은 문제를 \\(O(N^2)\\)으로 푼 다음 구사과님의 \\(O(NlogN)\\) 코드를 보고 공부해서 푼 거라 순수하게 저 혼자서 구현한 문제는 아니었습니다만 이번 문제는 저 혼자서 처음부터 끝까지 구현할 수 있었습니다.

사용 알고리즘 :
- Geometry
- memoization
- binary search

언뜻 직접 계산하라고 하면 못 할 것도 없어보이지만, 이걸 어떻게 컴퓨터한테 시킬 수 있을까요?
하나하나 생각해봅니다.

## 기하학

다각형의 넓이는 세 점이 주어졌을 때 외적으로 구할 수 있습니다. 넓이를 구하는 문제이므로 외적은 계속해서 사용하게 되겠군요. 외적만이 아니라 점들로 할 수 있는 연산은 다 할 것 같습니다. 연산자 오버로딩을 해줍니다.

{% highlight cpp %}
//C++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <algorithm>
#include <cmath>
typedef long long ll;
typedef long double ld;
const int LEN = 1e5 + 1;
//const ld TOL = 1e-7;
int N, M, Q;
ll memo_n[LEN]{ 0 }, memo_m[LEN]{ 0 };

//bool z(ld x) { return std::abs(x) < TOL; }
struct Pos {
    ll x, y;
    Pos(ll X, ll Y) : x(X), y(Y) {}
    Pos() : x(0), y(0) {}
    bool operator == (const Pos& p) const { return x == p.x && y == p.y; }
    bool operator < (const Pos& p) const { return x == p.x ? y < p.y : x < p.x; }
    Pos operator + (const Pos& p) const { return { x + p.x, y + p.y }; }
    Pos operator - (const Pos& p) const { return { x - p.x, y - p.y }; }
    Pos operator * (const ll& n) const { return { x * n, y * n }; }
    Pos operator / (const ll& n) const { return { x / n, y / n }; }
    ll operator * (const Pos& p) const { return { x * p.x + y * p.y }; }
    ll operator / (const Pos& p) const { return { x * p.y - y * p.x }; }
    Pos& operator += (const Pos& p) { x += p.x; y += p.y; return *this; }
    Pos& operator -= (const Pos& p) { x -= p.x; y -= p.y; return *this; }
    Pos& operator *= (const ll& scale) { x *= scale; y *= scale; return *this; }
    Pos& operator /= (const ll& scale) { x /= scale; y /= scale; return *this; }
    Pos operator ~ () const { return { -y, x }; }
    ll operator ! () const { return x * y; }
    ld mag() const { return hypot(x, y); }
} NH[LEN], MH[LEN], seq[LEN]; const Pos O = { 0, 0 };
struct Info { ll area, l, r; };
struct Query { int x; ld area; } q[LEN];
{% endhighlight %}

대소 비교, 점끼리 더하고 뺴기, 점 확대, 축소, 외적, 내적 등을 구현합니다.
바깥껍질 \\(NH\\), 안쪽껍질 \\(MH\\), 촛불의 위치 - 쿼리를 담아줄 \\(seq\\), 넓이를 구할 때 쓸 원점 \\(O\\)를 선언해줍니다.
실수 연산을 최소한으로 하기 위해 넓이와 좌, 우 점 위치를 기억할 구조체 \\(Info\\), 마지막에 출력할 답들을 넣을 구조체 \\(Query\\)까지 만들어줍니다.

{% highlight cpp %}
ll cross(const Pos& d1, const Pos& d2, const Pos& d3) { return (d2 - d1) / (d3 - d2); }
ll dot(const Pos& d1, const Pos& d2, const Pos& d3) { return (d2 - d1) * (d3 - d2); }
int ccw(const Pos& d1, const Pos& d2, const Pos& d3) {
    ll ret = cross(d1, d2, d3);
    return !ret ? 0 : ret > 0 ? 1 : -1;
}
bool on_seg(const Pos& d1, const Pos& d2, const Pos& d3) {
    return !ccw(d1, d2, d3) && dot(d1, d3, d2) >= 0;
}
{% endhighlight %}

외적, 내적, 선분 위 점 판정 등을 구현합니다.
여기까지는 기하 문제를 풀 때 필수적으로 구현해야하는 함수들입니다.

이제 본론으로 들어갑니다. 쿼리는 10만회이고, 다각형들의 정점도 10만개입니다. 그걸 다 일일이 뒤적거리고 있으면 컴퓨터 터집니다. 여기서 생각해볼 점은 다각형 두 개가 모두 볼록다각형이라는 것이며, 볼록다각형은 볼록성 때문에 이분 탐색이 가능하다는 특성이 있습니다. 또한 그림자가 될 다각형의 넓이는 가방끈 공식에 의해 `cross(O, H[i], H[i + 1])`의 누적합을 미리 \\(O(N)\\)에 전처리해두면 매 쿼리마다 \\(O(1)\\)에 구할 수 있습니다. ~~넓이를 누적합으로 구할 수 있는 건 어쩌다 생각난거고 그걸 가방끈 공식이라고 부르는 건 형한테서 들었습니다. 저도 잘 모릅니다.~~

## 누적합

{% highlight cpp %}
void get_area_memo(Pos H[], ll memo[], const int& sz) {
    memo[0] = 0;
    for (int i = 0; i < sz; i++) {
        Pos cur = H[i], nxt = H[(i + 1) % sz];
        memo[i + 1] = cross(O, cur, nxt) + memo[i];//memo[sz] == convex hull's area
    }
    return;
}
{% endhighlight %}

누적합 배열을 만들어준 후 `memo[0]`은 0, `memo[N]`은 다각형의 넓이를 기록해두도록 인덱스를 잘 조절해서 누적합을 구해줍니다.
다시 넓이를 구할 때는 두 점의 인덱스를 구하고 큰 인덱스의 넓이에서 작은 인덱스의 넓이를 빼고, 좌우를 판단한 후 `cross(O, H[i1], H[i2])`를 더해주면 잘려있는 다각형을 얻을 수 있습니다.

## 이분 탐색

이 문제의 핵심이자 이제부터 열불나게 돌려야하는 이분 탐색입니다.
우선 내부점 판정은 예전에 풀었던 [백준 20670 미스테리 싸인](https://www.acmicpc.net/problem/20670)의 함수를 사용합니다.

{% highlight cpp %}
int inner_check_bi_search(Pos H[], const int& sz, const Pos& p) {
    if (sz < 3 || cross(H[0], H[1], p) < 0 || cross(H[0], H[sz - 1], p) > 0) return -1;
    if (on_seg(H[0], H[1], p) || on_seg(H[0], H[sz - 1], p)) return 0;
    int s = 0, e = sz - 1, m;
    while (s + 1 < e) {
        m = s + e >> 1;
        if (cross(H[0], H[m], p) > 0) s = m;
        else e = m;
    }
    if (cross(H[s], H[e], p) > 0) return 1;
    else if (on_seg(H[s], H[e], p)) return 0;
    else return -1;
}
{% endhighlight %}

완전히 들어오는 점은 1, 테두리에 걸친 점은 0, 밖에 있는 점은 -1을 반환합니다.

{% highlight cpp %}
Info find_tangent_bi_search(Pos H[], const int& sz, const Pos& p) {
    int i1{ 0 }, i2{ 0 };
    int ccw1 = ccw(p, H[0], H[1]), ccwN = ccw(p, H[0], H[sz - 1]);
    if (ccw1 * ccwN >= 0) {
        i1 = 0;
        if (!ccw1 && dot(p, H[1], H[0]) > 0) i1 = 1;
        if (!ccwN && dot(p, H[sz - 1], H[0]) > 0) i1 = sz - 1;
        int s = 0, e = sz - 1, m;
        if (!ccw1) s += 1;
        if (!ccwN) e -= 1;
        bool f = ccw(p, H[s], H[s + 1]) >= 0;
        while (s < e) {
            m = s + e >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s = m + 1;
            else e = m;
        }
        i2 = s;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }
    else {
        //divide hull
        int s = 0, e = sz - 1, k, m;
        bool f = ccw1 > 0 && ccwN < 0;//if H[k] is between H[0] && p
        while (s + 1 < e) {
            k = s + e >> 1;
            int CCW = ccw(H[0], H[k], p);
            if (!f) CCW *= -1;//normailze
            if (CCW > 0) s = k;
            else e = k;
        }

        //search lower hull
        int s1 = 0, e1 = s;
        while (s1 < e1) {
            m = s1 + e1 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s1 = m + 1;
            else e1 = m;
        }
        i1 = s1;
        if (!ccw(p, H[i1], H[(i1 + 1) % sz]) && dot(p, H[(i1 + 1) % sz], H[i1]) > 0) i1 = (i1 + 1) % sz;

        //search upper hull
        int s2 = e, e2 = sz - 1;
        while (s2 < e2) {
            m = s2 + e2 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) < 0) s2 = m + 1;
            else e2 = m;
        }
        i2 = s2;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }    
    if (i2 < i1) std::swap(i2, i1);//normalize
    return { 0, i2, i1 };
}
{% endhighlight %}

이제 바깥에 있는 점으로부터의 두 접선을 그었을 때 접점을 찾아보도록 하겠습니다.

일단 접선을 찾기 이전에 이분 탐색을 돌릴 수 없는 반례인, 접점이 `H[0]`인 경우를 예외 처리 해줍니다.  예외 처리를 해주는 이유는 껍질이 두 개로 갈라지지 않기 때문입니다.

{% highlight cpp %}
    int i1{ 0 }, i2{ 0 };
    int ccw1 = ccw(p, H[0], H[1]), ccwN = ccw(p, H[0], H[sz - 1]);
    if (ccw1 * ccwN >= 0) {
        i1 = 0;
        if (!ccw1 && dot(p, H[1], H[0]) > 0) i1 = 1;
        if (!ccwN && dot(p, H[sz - 1], H[0]) > 0) i1 = sz - 1;
        int s = 0, e = sz - 1, m;
        if (!ccw1) s += 1;
        if (!ccwN) e -= 1;
        bool f = ccw(p, H[s], H[s + 1]) >= 0;
        while (s < e) {
            m = s + e >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s = m + 1;
            else e = m;
        }
        i2 = s;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }
{% endhighlight %}

![tan1](/assets/images/2024-02-05-candle/18190_1.jpg)

먼저 내부점 판정처럼 점 `p` 와 점 `H[0]`의 연장선이 가르는 선분을 찾아줍니다. 그리고 해당 반직선을 기준으로 껍질을 상대적 위와 아래로 나눈 후 양 깝질에 대해 이분 탐색을 돌려 양쪽에 대해 접점을 찾아줍니다. 이 때 점이 원점 `O`와 `H[0]`에 대해 어느 위치에 있는지에 따라 외적의 값이 반전되므로 빙향을 보고 정규화해줍니다.

![tan2](/assets/images/2024-02-05-candle/18190_2.jpg)

{% highlight cpp %}
    else {
        //divide hull
        int s = 0, e = sz - 1, k, m;
        bool f = ccw1 > 0 && ccwN < 0;//if H[k] is between H[0] && p
        while (s + 1 < e) {
            k = s + e >> 1;
            int CCW = ccw(H[0], H[k], p);
            if (!f) CCW *= -1;//normailze
            if (CCW > 0) s = k;
            else e = k;
        }

        //search lower hull
        int s1 = 0, e1 = s;
        while (s1 < e1) {
            m = s1 + e1 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s1 = m + 1;
            else e1 = m;
        }
        i1 = s1;
        if (!ccw(p, H[i1], H[(i1 + 1) % sz]) && dot(p, H[(i1 + 1) % sz], H[i1]) > 0) i1 = (i1 + 1) % sz;

        //search upper hull
        int s2 = e, e2 = sz - 1;
        while (s2 < e2) {
            m = s2 + e2 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) < 0) s2 = m + 1;
            else e2 = m;
        }
        i2 = s2;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }    
    if (i2 < i1) std::swap(i2, i1);//normalize
    return { 0, i2, i1 };
}
{% endhighlight %}

![tan3](/assets/images/2024-02-05-candle/18190_3.jpg)

![tan4](/assets/images/2024-02-05-candle/18190_4.jpg)

`ccw`를 판정해 마지막까지 정규화를 해주고 두 접점의 인덱스를 반환합니다. 이로써 두 접점의 위치를 찾았습니다.

{% highlight cpp %}
Info get_inner_area(Pos H[], ll memo[], const int& sz, const Pos& p) {
    Info tangent = find_tangent_bi_search(H, sz, p);
    ll i1 = tangent.r, i2 = tangent.l;
    ll tri = cross(O, H[i1], H[i2]);
    ll area = memo[i2] - memo[i1] - tri;
    if (cross(p, H[i1], H[i2]) < 0) area = memo[sz] - area, std::swap(i1, i2);//normalize
    area += std::abs(cross(p, H[i1], H[i2]));
    return { area, i2, i1 };
}
{% endhighlight %}

![area1](/assets/images/2024-02-05-candle/18190_10.jpg)

누적합으로 구해뒀던 넓이들을 활용해 안쪽 다각형의 넓이를 구해줍니다.

<br></br>

다음은 바깥 껍질과 그림자의 접점을 찾아봅시다.

{% highlight cpp %}
ld find_inx_get_area_bi_search(Pos H_in[], ll memo_in[], const int& sz_in, Pos H_out[], ll memo_out[], const int& sz_out, const Pos& p) {
    Info info = get_inner_area(H_in, memo_in, sz_in, p);
    Pos vr = H_in[info.r], vl = H_in[info.l];
    int ir, il;
    ld wing_r{ 0 }, wing_l{ 0 };

    //divide hull
    int s = 0, e = sz_out - 1, k, m;
    while (s + 1 < e) {
        k = s + e >> 1;
        int CCW = ccw(H_out[0], H_out[k], p);
        if (CCW > 0) s = k;
        else e = k;
    }
    Pos S = H_out[s], E = H_out[e];
 
    //find r-intersection
    int sr{ 0 }, er{ 0 };
    Pos SR, ER;
    if (ccw(p, S, vr) >= 0 && ccw(p, E, vr) <= 0) sr = s, er = e;//if vr is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vr) > 0) sr = e, er = sz_out;
        if (ccw(H_out[0], p, vr) < 0) sr = 0, er = s;
        while (sr + 1 < er) {
            m = sr + er >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vr);
            if (CCW > 0) sr = m;
            else er = m;
        }
    }
    SR = H_out[sr % sz_out], ER = H_out[er % sz_out];
    ir = er % sz_out;
    ll trir = std::abs(cross(p, SR, ER));
    ll ar = std::abs(cross(p, vr, SR));
    ll br = std::abs(cross(p, vr, ER));
    wing_r = trir * (ld)br / (ar + br);
    if (!cross(p, vr, H_out[er % sz_out])) wing_r = 0;

    //find l-intersection
    int sl{ 0 }, el{ 0 };
    Pos SL, EL;
    if (ccw(p, S, vl) >= 0 && ccw(p, E, vl) <= 0) sl = s, el = e;//if vl is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vl) > 0) sl = e, el = sz_out;
        if (ccw(H_out[0], p, vl) < 0) sl = 0, el = s;
        while (sl + 1 < el) {
            m = sl + el >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vl);
            if (CCW > 0) sl = m;
            else el = m;
        }
    }
    SL = H_out[sl % sz_out], EL = H_out[el % sz_out];
    il = sl % sz_out;
    ll tril = std::abs(cross(p, SL, EL));
    ll al = std::abs(cross(p, vl, SL));
    ll bl = std::abs(cross(p, vl, EL));
    wing_l = tril * (ld)al / (al + bl);
    if (!cross(p, vl, H_out[sl % sz_out])) wing_l = 0;
   
    //get_shadow
    ld area{ 0 };
    if (sr == sl) {//if 2 intersections on the same segment
        area = -(ld)(info.area + std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    else {
        bool f = ir > il;
        if (ir > il) std::swap(ir, il);//normalize
        ll tri = cross(O, H_out[ir], H_out[il]);
        ll tmp = memo_out[il] - memo_out[ir] - tri;
        if (f) tmp = memo_out[sz_out] - tmp;
        area = -(ld)(info.area - tmp - std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    return area * .5;
}
{% endhighlight %}

안쪽 껍질에서 했던 것처럼 일단 다각형을 반으로 갈라줍니다.

![sh1](/assets/images/2024-02-05-candle/18190_5.jpg)

{% highlight cpp %}
    Info info = get_inner_area(H_in, memo_in, sz_in, p);
    Pos vr = H_in[info.r], vl = H_in[info.l];
    int ir, il;
    ld wing_r{ 0 }, wing_l{ 0 };

    //divide hull
    int s = 0, e = sz_out - 1, k, m;
    while (s + 1 < e) {
        k = s + e >> 1;
        int CCW = ccw(H_out[0], H_out[k], p);
        if (CCW > 0) s = k;
        else e = k;
    }
    Pos S = H_out[s], E = H_out[e];
{% endhighlight %}

![sh2](/assets/images/2024-02-05-candle/18190_6.jpg)

위아래로 갈라진 두 껍질에 대해 각각 점이 어느 위치에 있는지를 파악해준 후, 갈라진 양쪽 다각형을 제외한 남은 삼각형에 점점이 있는 경우를 예외 처리 해주고 다른 자각형 중 하나에 있을 경우 이분 탐색을 돌려줍니다. 이렇게 구한 오른쪽과 왼쪽 접점이 위치한 범위로부터 그림자의 경계와 가장 가까운 점 2개를 찾고, 다각형의 변과 그림자의 경계가 만나는 양쪽 삼각형 (변수명을 날개라고 지었습니다)의 넓이도 바로 찾아줍니다. 양 날개의 넓이는 외적 간의 비례식으로 구하면 최소한의 실수 연산으로 구할 수 있습니다. 여기서 또 그림자의 경계가 바깥 껍질의 접점과 같은 경우를 예외 처리 해줍니다.

{% highlight cpp %}
    //find r-intersection
    int sr{ 0 }, er{ 0 };
    Pos SR, ER;
    if (ccw(p, S, vr) >= 0 && ccw(p, E, vr) <= 0) sr = s, er = e;//if vr is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vr) > 0) sr = e, er = sz_out;
        if (ccw(H_out[0], p, vr) < 0) sr = 0, er = s;
        while (sr + 1 < er) {
            m = sr + er >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vr);
            if (CCW > 0) sr = m;
            else er = m;
        }
    }
    SR = H_out[sr % sz_out], ER = H_out[er % sz_out];
    ir = er % sz_out;
    ll trir = std::abs(cross(p, SR, ER));
    ll ar = std::abs(cross(p, vr, SR));
    ll br = std::abs(cross(p, vr, ER));
    wing_r = trir * (ld)br / (ar + br);
    if (!cross(p, vr, H_out[er % sz_out])) wing_r = 0;

    //find l-intersection
    int sl{ 0 }, el{ 0 };
    Pos SL, EL;
    if (ccw(p, S, vl) >= 0 && ccw(p, E, vl) <= 0) sl = s, el = e;//if vl is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vl) > 0) sl = e, el = sz_out;
        if (ccw(H_out[0], p, vl) < 0) sl = 0, el = s;
        while (sl + 1 < el) {
            m = sl + el >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vl);
            if (CCW > 0) sl = m;
            else el = m;
        }
    }
    SL = H_out[sl % sz_out], EL = H_out[el % sz_out];
    il = sl % sz_out;
    ll tril = std::abs(cross(p, SL, EL));
    ll al = std::abs(cross(p, vl, SL));
    ll bl = std::abs(cross(p, vl, EL));
    wing_l = tril * (ld)al / (al + bl);
    if (!cross(p, vl, H_out[sl % sz_out])) wing_l = 0;
{% endhighlight %}

![sh3](/assets/images/2024-02-05-candle/18190_7.jpg)

이로써 그림자의 넓이를 구하는 데 필요한 값들은 다 찾았습니다.

나머지는 누적합으로 빠르게 구해줍니다.

![sh4](/assets/images/2024-02-05-candle/18190_8.jpg)

안쪽 껍질은 위에서도 구했지만 함수를 다시 쓰자면 아래와 같고

{% highlight cpp %}
Info get_inner_area(Pos H[], ll memo[], const int& sz, const Pos& p) {
    Info tangent = find_tangent_bi_search(H, sz, p);
    ll i1 = tangent.r, i2 = tangent.l;
    ll tri = cross(O, H[i1], H[i2]);
    ll area = memo[i2] - memo[i1] - tri;
    if (cross(p, H[i1], H[i2]) < 0) area = memo[sz] - area, std::swap(i1, i2);//normalize
    area += std::abs(cross(p, H[i1], H[i2]));
    return { area, i2, i1 };
}
{% endhighlight %}

바깥 껍질에서 만나는 그림자의 넓이는 아래청럼 구합니다. 두 그림자의 경계가 한 변에서 만나는 경우를 예외 처리 해줍니다.

{% highlight cpp %}   
    //get_shadow
    ld area{ 0 };
    if (sr == sl) {//if 2 intersections on the same segment
        area = -(ld)(info.area + std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    else {
        bool f = ir > il;
        if (ir > il) std::swap(ir, il);//normalize
        ll tri = cross(O, H_out[ir], H_out[il]);
        ll tmp = memo_out[il] - memo_out[ir] - tri;
        if (f) tmp = memo_out[sz_out] - tmp;
        area = -(ld)(info.area - tmp - std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    return area * .5;
}
{% endhighlight %}

![shadow](/assets/images/2024-02-05-candle/18190_9.jpg)

이분 탐섹으로 시작해서 이분 탐색으로 끝나는 문제입니다.

{% highlight cpp %}
void query(const int& i) {
    Pos candle = seq[i];
    if (inner_check_bi_search(MH, M, candle) > -1) q[i].x = 1;
    else if (inner_check_bi_search(NH, N, candle) < 1) q[i].x = -1;
    else q[i].x = 0, q[i].area = find_inx_get_area_bi_search(MH, memo_m, M, NH, memo_n, N, candle);
    return;
}
void query_print() {
    for (int i = 0; i < Q; i++) {
        if (q[i].x == 0) std::cout << q[i].area << "\n";
        else if (q[i].x == 1) std::cout << "IN\n";
        else if (q[i].x == -1) std::cout << "OUT\n";
    }
    return;
}
void init() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cout << std::fixed;
    std::cout.precision(7);
    std::cin >> N >> M >> Q;
    for (int i = 0; i < N; i++) std::cin >> NH[i].x >> NH[i].y;
    for (int j = 0; j < M; j++) std::cin >> MH[j].x >> MH[j].y;
    for (int k = 0; k < Q; k++) std::cin >> seq[k].x >> seq[k].y;
    get_area_memo(NH, memo_n, N);
    get_area_memo(MH, memo_m, M);
    return;
}
void solve() { init(); for (int i = 0; i < Q; i++) query(i); query_print(); return; }
int main() { solve(); return 0; }//boj18190
{% endhighlight %}

쿼리를 받고 내부 점 판정을 해준 뒤 넓이가 발생할 경우 이분 탐색으로 넓이를 찾아줍니다. 쿼리를 10만번 돌아줍시다.

전체 코드

{% highlight cpp %}
//C++
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <algorithm>
#include <cmath>
typedef long long ll;
typedef long double ld;
const int LEN = 1e5 + 1;
//const ld TOL = 1e-7;
int N, M, Q;
ll memo_n[LEN]{ 0 }, memo_m[LEN]{ 0 };

//bool z(ld x) { return std::abs(x) < TOL; }
struct Pos {
    ll x, y;
    Pos(ll X, ll Y) : x(X), y(Y) {}
    Pos() : x(0), y(0) {}
    bool operator == (const Pos& p) const { return x == p.x && y == p.y; }
    bool operator < (const Pos& p) const { return x == p.x ? y < p.y : x < p.x; }
    Pos operator + (const Pos& p) const { return { x + p.x, y + p.y }; }
    Pos operator - (const Pos& p) const { return { x - p.x, y - p.y }; }
    Pos operator * (const ll& n) const { return { x * n, y * n }; }
    Pos operator / (const ll& n) const { return { x / n, y / n }; }
    ll operator * (const Pos& p) const { return { x * p.x + y * p.y }; }
    ll operator / (const Pos& p) const { return { x * p.y - y * p.x }; }
    Pos& operator += (const Pos& p) { x += p.x; y += p.y; return *this; }
    Pos& operator -= (const Pos& p) { x -= p.x; y -= p.y; return *this; }
    Pos& operator *= (const ll& scale) { x *= scale; y *= scale; return *this; }
    Pos& operator /= (const ll& scale) { x /= scale; y /= scale; return *this; }
    Pos operator ~ () const { return { -y, x }; }
    ll operator ! () const { return x * y; }
    ld mag() const { return hypot(x, y); }
} NH[LEN], MH[LEN], seq[LEN]; const Pos O = { 0, 0 };
struct Info { ll area, l, r; };
struct Query { int x; ld area; } q[LEN];
ll cross(const Pos& d1, const Pos& d2, const Pos& d3) { return (d2 - d1) / (d3 - d2); }
ll dot(const Pos& d1, const Pos& d2, const Pos& d3) { return (d2 - d1) * (d3 - d2); }
int ccw(const Pos& d1, const Pos& d2, const Pos& d3) {
    ll ret = cross(d1, d2, d3);
    return !ret ? 0 : ret > 0 ? 1 : -1;
}
bool on_seg(const Pos& d1, const Pos& d2, const Pos& d3) {
    return !ccw(d1, d2, d3) && dot(d1, d3, d2) >= 0;
}
void get_area_memo(Pos H[], ll memo[], const int& sz) {
    memo[0] = 0;
    for (int i = 0; i < sz; i++) {
        Pos cur = H[i], nxt = H[(i + 1) % sz];
        memo[i + 1] = cross(O, cur, nxt) + memo[i];//memo[sz] == convex hull's area
    }
    return;
}
int inner_check_bi_search(Pos H[], const int& sz, const Pos& p) {
    if (sz < 3 || cross(H[0], H[1], p) < 0 || cross(H[0], H[sz - 1], p) > 0) return -1;
    if (on_seg(H[0], H[1], p) || on_seg(H[0], H[sz - 1], p)) return 0;
    int s = 0, e = sz - 1, m;
    while (s + 1 < e) {
        m = s + e >> 1;
        if (cross(H[0], H[m], p) > 0) s = m;
        else e = m;
    }
    if (cross(H[s], H[e], p) > 0) return 1;
    else if (on_seg(H[s], H[e], p)) return 0;
    else return -1;
}
Info find_tangent_bi_search(Pos H[], const int& sz, const Pos& p) {
    int i1{ 0 }, i2{ 0 };
    int ccw1 = ccw(p, H[0], H[1]), ccwN = ccw(p, H[0], H[sz - 1]);
    if (ccw1 * ccwN >= 0) {
        i1 = 0;
        if (!ccw1 && dot(p, H[1], H[0]) > 0) i1 = 1;
        if (!ccwN && dot(p, H[sz - 1], H[0]) > 0) i1 = sz - 1;
        int s = 0, e = sz - 1, m;
        if (!ccw1) s += 1;
        if (!ccwN) e -= 1;
        bool f = ccw(p, H[s], H[s + 1]) >= 0;
        while (s < e) {
            m = s + e >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s = m + 1;
            else e = m;
        }
        i2 = s;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }
    else {
        //divide hull
        int s = 0, e = sz - 1, k, m;
        bool f = ccw1 > 0 && ccwN < 0;//if H[k] is between H[0] && p
        while (s + 1 < e) {
            k = s + e >> 1;
            int CCW = ccw(H[0], H[k], p);
            if (!f) CCW *= -1;//normailze
            if (CCW > 0) s = k;
            else e = k;
        }

        //search lower hull
        int s1 = 0, e1 = s;
        while (s1 < e1) {
            m = s1 + e1 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) > 0) s1 = m + 1;
            else e1 = m;
        }
        i1 = s1;
        if (!ccw(p, H[i1], H[(i1 + 1) % sz]) && dot(p, H[(i1 + 1) % sz], H[i1]) > 0) i1 = (i1 + 1) % sz;

        //search upper hull
        int s2 = e, e2 = sz - 1;
        while (s2 < e2) {
            m = s2 + e2 >> 1;
            Pos p1 = p, cur = H[m], nxt = H[(m + 1) % sz];
            if (!f) std::swap(p1, cur);//normalize
            if (ccw(p1, cur, nxt) < 0) s2 = m + 1;
            else e2 = m;
        }
        i2 = s2;
        if (!ccw(p, H[i2], H[(i2 + 1) % sz]) && dot(p, H[(i2 + 1) % sz], H[i2]) > 0) i2 = (i2 + 1) % sz;
    }    
    if (i2 < i1) std::swap(i2, i1);//normalize
    return { 0, i2, i1 };
}
Info get_inner_area(Pos H[], ll memo[], const int& sz, const Pos& p) {
    Info tangent = find_tangent_bi_search(H, sz, p);
    ll i1 = tangent.r, i2 = tangent.l;
    ll tri = cross(O, H[i1], H[i2]);
    ll area = memo[i2] - memo[i1] - tri;
    if (cross(p, H[i1], H[i2]) < 0) area = memo[sz] - area, std::swap(i1, i2);//normalize
    area += std::abs(cross(p, H[i1], H[i2]));
    return { area, i2, i1 };
}
ld find_inx_get_area_bi_search(Pos H_in[], ll memo_in[], const int& sz_in, Pos H_out[], ll memo_out[], const int& sz_out, const Pos& p) {
    Info info = get_inner_area(H_in, memo_in, sz_in, p);
    Pos vr = H_in[info.r], vl = H_in[info.l];
    int ir, il;
    ld wing_r{ 0 }, wing_l{ 0 };

    //divide hull
    int s = 0, e = sz_out - 1, k, m;
    while (s + 1 < e) {
        k = s + e >> 1;
        int CCW = ccw(H_out[0], H_out[k], p);
        if (CCW > 0) s = k;
        else e = k;
    }
    Pos S = H_out[s], E = H_out[e];

    //find r-intersection
    int sr{ 0 }, er{ 0 };
    Pos SR, ER;
    if (ccw(p, S, vr) >= 0 && ccw(p, E, vr) <= 0) sr = s, er = e;//if vr is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vr) > 0) sr = e, er = sz_out;
        if (ccw(H_out[0], p, vr) < 0) sr = 0, er = s;
        while (sr + 1 < er) {
            m = sr + er >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vr);
            if (CCW > 0) sr = m;
            else er = m;
        }
    }
    SR = H_out[sr % sz_out], ER = H_out[er % sz_out];
    ir = er % sz_out;
    ll trir = std::abs(cross(p, SR, ER));
    ll ar = std::abs(cross(p, vr, SR));
    ll br = std::abs(cross(p, vr, ER));
    wing_r = trir * (ld)br / (ar + br);
    if (!cross(p, vr, H_out[er % sz_out])) wing_r = 0;

    //find l-intersection
    int sl{ 0 }, el{ 0 };
    Pos SL, EL;
    if (ccw(p, S, vl) >= 0 && ccw(p, E, vl) <= 0) sl = s, el = e;//if vl is in p-S-E tri.
    else {
        if (ccw(H_out[0], p, vl) > 0) sl = e, el = sz_out;
        if (ccw(H_out[0], p, vl) < 0) sl = 0, el = s;
        while (sl + 1 < el) {
            m = sl + el >> 1;
            int CCW = ccw(p, H_out[m % sz_out], vl);
            if (CCW > 0) sl = m;
            else el = m;
        }
    }
    SL = H_out[sl % sz_out], EL = H_out[el % sz_out];
    il = sl % sz_out;
    ll tril = std::abs(cross(p, SL, EL));
    ll al = std::abs(cross(p, vl, SL));
    ll bl = std::abs(cross(p, vl, EL));
    wing_l = tril * (ld)al / (al + bl);
    if (!cross(p, vl, H_out[sl % sz_out])) wing_l = 0;

    //DEBUG
    //std::cout << "in R: " << info.r << " in L: " << info.l << " out R: " << ir << " out L: " << il << "\n";
    //std::cout << "wing R: " << wing_r << " wing L : " << wing_l << "\n";
    //std::cout << "wing R: " << trir * (ld)ar / (ar + br) << " wing L : " << tril * (ld)bl / (al + bl) << "\n";

    //std::cout << "inner: " << info.area << "\n";    
    //get_shadow
    ld area{ 0 };
    if (sr == sl) {//if 2 intersections on the same segment
        area = -(ld)(info.area + std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    else {
        bool f = ir > il;
        if (ir > il) std::swap(ir, il);//normalize
        ll tri = cross(O, H_out[ir], H_out[il]);
        ll tmp = memo_out[il] - memo_out[ir] - tri;
        if (f) tmp = memo_out[sz_out] - tmp;
        area = -(ld)(info.area - tmp - std::abs(cross(p, H_out[ir], H_out[il]))) + (wing_r + wing_l);
    }
    return area * .5;
}
void query(const int& i) {
    Pos candle = seq[i];
    if (inner_check_bi_search(MH, M, candle) > -1) q[i].x = 1;
    else if (inner_check_bi_search(NH, N, candle) < 1) q[i].x = -1;
    else q[i].x = 0, q[i].area = find_inx_get_area_bi_search(MH, memo_m, M, NH, memo_n, N, candle);
    return;
}
void query_print() {
    for (int i = 0; i < Q; i++) {
        if (q[i].x == 0) std::cout << q[i].area << "\n";
        else if (q[i].x == 1) std::cout << "IN\n";
        else if (q[i].x == -1) std::cout << "OUT\n";
    }
    return;
}
void init() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cout << std::fixed;
    std::cout.precision(7);
    std::cin >> N >> M >> Q;
    for (int i = 0; i < N; i++) std::cin >> NH[i].x >> NH[i].y;
    for (int j = 0; j < M; j++) std::cin >> MH[j].x >> MH[j].y;
    for (int k = 0; k < Q; k++) std::cin >> seq[k].x >> seq[k].y;
    get_area_memo(NH, memo_n, N);
    get_area_memo(MH, memo_m, M);
    return;
}
void solve() { init(); for (int i = 0; i < Q; i++) query(i); query_print(); return; }
int main() { solve(); return 0; }//boj18190
{% endhighlight %}
