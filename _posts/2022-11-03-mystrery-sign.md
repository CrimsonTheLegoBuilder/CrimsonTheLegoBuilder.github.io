---
layout: post
title: 백준 20670 미스테리 싸인
date: 2023-11-03 15:00:00 +0900
categories:
- PS
- Geometry
- Convex hull
- Point in convex poligon
- Binary search
tags:
- PS
- Geometry
- Convex hull
- Point in convex poligon
- Binary search
description: 이분 탐색으로 볼록 다각형 내부 점 판정을 O(log N)에 시행한다.
usemathjax: true
---


{% include rate.html image_path="/assets/images/rate/P3.svg" url="https://www.acmicpc.net/problem/20670" discription="20670 미스테리 싸인"%}

두 개의 볼록다각형이 주어지고, 싸인을 구성하는 여러 개의 점이 주어집니다. 점은 바깥쪽 볼록 다각형 안에 존재해야하며 동시에 안쪽 볼록 다각형 밖에 존재해야 합니다.

우선 볼록 다각형이 어떻게 표현되는지부터 봅시다. 모든 점은 반시계방향 순서로 주어진다고 합니다. 컴퓨터 입장에서 점들의 좌표는 그저 의미 없는 1과 0의 덩어리일 뿐입니다. 점들이 의미하는 것은 온전히 사람의 입장에서 정의됩니다. 점들의 순서가 꼬이게 되면 점들은 볼록 다각형을 의미하지 못하게 되므로 볼록 다각형을 표현하는데 있어서는 순서가 매우 중요합니다.

이제 내부 점 판정을 생각해보겠습니다. 볼록 다각형 내부 점 판정은 \\(O(N)\\) 에 확인하는 방법과 \\(O(log N)\\) 에 확인하는 방법이 있습니다. \\(O(N)\\) 에 확인하는 방법은 앞에서 다룬 `CCW`를 다각형을 이루는 모든 변과 목표 점에 대해 확인해서 모든 CCW가 참이 되는 것을 확인하는 것입니다. 직관적이며 구현도 간단하지만 모든 변에 대해서 확인해야하므로 일종의 브루트포스가 됩니다. \\(O(log N)\\) 에 확인하는 방법은 이분 탐색으로 볼록 다각형을 반 씩 갈라가며 점이 존재하는 변의 범위를 확정 짓고 안에 존재하는지를 확인하는 것입니다.

이 문제에서 주어지는 두 다각형을 이루는 점 \\(N\\) 과 \\(M\\) 의 크기는 각각 10,000개까지 가능하며 점의 수는 300,000개입니다. 저걸 일일이 \\(O(N)\\) 과 \\(O(M)\\) 에 구하고 있으면 시간 초과입니다. 따라서 \\(O(log N)\\) 에 모든 점의 내부 점 판정을 할 수 있어야 시간 안에 AC를 받을 수 있습니다.

먼저 점을 구성하는 구조체 `Pos`를 만들어줍니다. `std::vector`에서 제공하는 `Pair` 구조체를 써도 됩니다. 정렬 순서와 ID를 부여하는 등의 기법을 쓰는데는 직접 구현한 구조체가 쓰기 편해서 만들어 쓰는 편을 선호합니다.

{% highlight cpp %}
#include <iostream>
#include <algorithm>
typedef long long ll;
const int LEN = 10'000;
int n, m, k, out;  //바깥쪽 다각형의 점 수 n, 안쪽 다각형의 점 수 m, 싸인의 점 수 k

struct Pos {
    ll x, y;
    bool operator < (const Pos& p) const { return (x == p.x ? y < p.y : x < p.x); }
    //두 점의 정렬 순서. 이 문제에서는 정렬을 하지 않아 기준이 쓰이지 않았다.
}N[LEN], M[LEN], K;  //바깥쪽 다각형 N, 안쪽 다각형 M, 싸인의 좌표 K
{% endhighlight %}

세 점으로 `CCW`를 판단하는 외적을 구현해줍니다.

{% highlight cpp %}
ll cross(const Pos& d1, const Pos& d2, const Pos& d3) {
    return (d2.x - d1.x) * (d3.y - d2.y) - (d2.y - d1.y) * (d3.x - d2.x);
}
{% endhighlight %}

이제 내부 점 판정을 하러 가봅시다.
볼록 다각형의 \\(0\\)번째 점을 고정으로 잡고, \\(1\\)번째와 \\(h - 1\\)번째 양쪽 점에 대해서 점이 변 밖에 나가있는지부터 판단합니다.

{% highlight cpp %}
bool I(const Pos& p, Pos H[], int h) {  //h = 볼록 다각형의 점의 개수
    if (h < 3 || cross(H[0], H[1], p) <= 0 || cross(H[0], H[h - 1], p) >= 0) return 0;
    //길이가 3 이하이면 볼록 다각형이 아니어서 내부 판정 자체가 불가능하다.
    //0-1 번 째 변 오른쪽에 있거나 0-(h-1) 번 째 변 왼쪽에 있으면 해당 점은 외부에 있다.
{% endhighlight %}

이제 이분 탐색으로 점이 0번 점으로부터 시작해 어떤 두 점 사이에 있는지를 판단합니다. 먼저 \\(s = 0 , e = h - 1\\) 로 설정해줍니다. \\(m\\)번째 점을 잡아준 후 `cross(H[0], H[m], p)`로 `CCW` (p가 왼쪽에 있는지, m이 오른쪽에 있는지)를 확인힙니다. p가 왼쪽에 있다면 `(CCW)` s부터 m까지의 점들은 더 이상 고려할 필요가 없어집니다. 반쪽을 날려줍시다. 판단 결과가 반대라면 반대쪽도 똑같이 날려줍니다. 이 과정을 s와 e의 차이가 1 만큼일 때까지 반복해줍니다.

![CCWin](/assets/images/2023-11-03-in/CCW_in.jpg)

{% highlight cpp %}
    int s = 0, e = h - 1, m;
    while (s + 1 < e) {
        m = s + e >> 1;
        if (cross(H[0], H[m], p) > 0) s = m;
        //외적 결과가 0보다 크다면 m 번째 점은 점 p보다 오른쪽에 있으므로
        //m 오른쪽인 s~m 사이 점들을 날려버린다.
        else e = m;
        //반대쪽도 대칭으로 시행
	}
{% endhighlight %}

점 p가 어떤 변의 범위 안에 위치하는지 알아냈습니다. 이제 남은 일은 점 s, e, p가 CCW를 형성하는지 알아내는 것입니다.

{% highlight cpp %}
    return cross(H[s], H[e], p) > 0;
}
{% endhighlight %}

이제 볼록 다각형 두 개를 입력받고 내부 점 판정을 돌려줍니다.

{% highlight cpp %}
int main() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cin >> n >> m >> k;
    for (int i = 0; i < n; i++) { std::cin >> N[i].x >> N[i].y; }
    for (int i = 0; i < m; i++) { std::cin >> M[i].x >> M[i].y; }
    for (int i = 0; i < k; i++) {
        std::cin >> K.x >> K.y;
        if (!I(K, N, n) || I(K, M, m)) out++;
        //N 밖에 있거나 M 안에 있다면 조건을 벗어난다.
    }
    if (!out) std::cout << "YES" << "\n";
    else std::cout << out << "\n";
    return 0;
}
{% endhighlight %}

BOJ 20670 미스테리 싸인 전체 코드

{% highlight cpp %}
#include <iostream>
#include <algorithm>
typedef long long ll;
const int LEN = 10'000;
int n, m, k, out;

struct Pos {
    ll x, y;
    bool operator < (const Pos& p) const { return (x == p.x ? y < p.y : x < p.x); }
}N[LEN], M[LEN], K;

ll cross(const Pos& d1, const Pos& d2, const Pos& d3) {
    return (d2.x - d1.x) * (d3.y - d2.y) - (d2.y - d1.y) * (d3.x - d2.x);
}
bool I(const Pos& p, Pos H[], int h) {
    if (h < 3 || cross(H[0], H[1], p) <= 0 || cross(H[0], H[h - 1], p) >= 0) return 0;
    int s = 0, e = h - 1, m;
    while (s + 1 < e) {
        m = s + e >> 1;
        if (cross(H[0], H[m], p) > 0) s = m;
        else e = m;
    }
    return cross(H[s], H[e], p) > 0;
}



int main() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cin >> n >> m >> k;
    for (int i = 0; i < n; i++) { std::cin >> N[i].x >> N[i].y; }
    for (int i = 0; i < m; i++) { std::cin >> M[i].x >> M[i].y; }
    for (int i = 0; i < k; i++) {
        std::cin >> K.x >> K.y;
        if (!I(K, N, n) || I(K, M, m)) out++;
    }
    if (!out) std::cout << "YES" << "\n";
    else std::cout << out << "\n";
    return 0;
}
{% endhighlight %}

메모리를 희생하고 싸인을 이루는 점을 미리 전부 받아 시간을 단축하는 방법도 있습니다.

{% highlight cpp %}
#include <iostream>
#include <algorithm>
typedef long long ll;
const int LEN = 10'000;
int n, m, k, out;

struct Pos { ll x, y; }N[LEN], M[LEN], K[LEN * 30];

ll cross(const Pos& d1, const Pos& d2, const Pos& d3) {
    return (d2.x - d1.x) * (d3.y - d2.y) - (d2.y - d1.y) * (d3.x - d2.x);   
}
bool I(const Pos& p, Pos H[], int h) {
    if (h < 3 || cross(H[0], H[1], p) <= 0 || cross(H[0], H[h - 1], p) >= 0) return 0;
    int s = 0, e = h - 1, m;
    while (s + 1 < e) {
        m = s + e >> 1;
        if (cross(H[0], H[m], p) > 0) s = m;
        else e = m;
    }
    return cross(H[s], H[e], p) > 0;
}



int main() {
    std::cin.tie(0)->sync_with_stdio(0);
    std::cout.tie(0);
    std::cin >> n >> m >> k;
    for (int i = 0; i < n; i++) { std::cin >> N[i].x >> N[i].y; }
    for (int i = 0; i < m; i++) { std::cin >> M[i].x >> M[i].y; }
    for (int i = 0; i < k; i++) { std::cin >> K[i].x >> K[i].y; }
    for (int i = 0; i < k; i++) { if (!I(K[i], N, n) || I(K[i], M, m)) out++; }
    if (!out) std::cout << "YES" << "\n";
    else std::cout << out << "\n";
    return 0;
}
`{% endhighlight %}