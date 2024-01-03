---
layout: post
title: 백준 3527 Jungle Outpost
date: 2024-01-03 15:30:00 +0900
categories:
- PS
tags:
- PS
description: BOJ 3527 Jungle Outpost
usemathjax: true
---

# BOJ 3527 Jungle Outpost

{% include rate.html image_path="/assets/images/rate/D1.svg" url="https://www.acmicpc.net/problem/3527" discription="3527 Jungle Outpost"%}

쌍둥이 형이 알고리즘 문제 풀이를 권유하면서 소개해줬던 기하학 문제 중 하나입니다. 맨 처음 봤을 때는 너나 풀어라 하면서 무시했던 문제였는데 결국 제가 먼저 풀고 처음으로 형한테 풀이와 알고리즘을 알려준 문제이기도 합니다.
사용된 알고리즘도 구현이 어려운 편에 속하는데다 쓰이는 알고리즘들을 다 알고 있음에도 풀이를 떠올리는데 시간이 꽤 걸린 문제입니다. 정작 풀이를 떠올리고 나서는 구현하는데까지 시간이 얼마 걸리지 않았고 한 번에 맞춘 문제이기도 합니다. 풀이만 알고 나면 꽤 간단한(?) 문제입니다.

사용 알고리즘 :
- Geometry
- Convex hull
- halfplane intersection
- binary search

## 기하학

## 볼록 껍질

일단 문제의 조건을 뜯어봅시다. 보호막 발생 장치는 볼록 다각형 모양으로 배치되어있고 파손된 장치는 보호막 발생 능력을 잃습니다. 따라서 적들은 최소한의 장치만을 파괴해 공격을 할 것이고 우리는 가장 적은 장치들만으로도 유지되는 보호막이 어느 수준인지를 알아야합니다.
장치가 아주 많고 원 모양으로 배치되어 있다고 가정해보겠습니다. 장치는 10만개가 있는데 하나 건너 하나씩 파괴한다고 한다면, 우리의 보호막은 아직 5만개의 장치와 함께 (거의) 건재할 것입니다. 반대로, 모든 장치 중 한 쪽에 치우친 장치들만 골라서 파괴한다면, 장치를 절반만 날리고도 보호되는 영역이 반으로 줍니다. 따라서 적은 어떻게든 한 줄로 붙어있는 장치들을 파괴하려 할 것이고, 우리는 한 쪽의 장치들이 무사하다는 가정 하에 보호되는 영역이 얼마나 오래 유지되는지 알아내야 합니다.
마치 카메라의 조리개가 모이면서 가운데의 렌즈가 가려지는 과정을 떠올려보면 이 문제가 직감적으로 외닿을 것입니다.

## 빈평면 교집합

절반의 장치들이 파괴되었다고 한다면 아직 남아있는 장치들로 생성되는 영역이 있을 것이고, 절반의 붙어있는 장치를 골랐을 때 각각의 영역들이 겹쳐져 생기는 영역은 반평면 교집합이 됩니다. 즉 이 문제는 장치를 몇 개까지 파괴했을 때 절대적으로 보호되는 영역이 없어지는지, 즉, 몇 칸을 건너뛰어 생기는 반평면들의 교집합이 생기지 않게 되는지를 묻는 문제가 됩니다.

## 이분 탐색

볼록 껍질을 건너뛰어 반평면 교집합이 생기는 걸 구하고, 아직 구역이 생긴다면 더 적게 파괴하는 경우는 무시하고, 생기지 않는다면 더 적게 파괴할 수 있는 수를 찾는 과정을 거쳐야하는데, 개형이 0 또는 1로 결정되므로 이분 탐색을 적용할 수 있는 형태가 됩니다.

{% highlight cpp %}
//C++
#include <iostream>
#include <algorithm>
#include <vector>
#include <deque>
#include <cmath>
typedef long long ll;
typedef double ld;
const int LEN = 50'000;
const ld TOL = 1e-8;
int N;

bool z(ld x) { return std::fabs(x) < TOL; }  // x == zero ?
struct Pos { ld x, y; }pos[LEN];
struct Line {
    ld vy, vx, c;  // a(vy) * x + b(-vx) * y - c == 0;
    bool operator < (const Line& l) const {
        bool f1 = z(vy) ? vx > 0 : vy > 0;
        bool f2 = z(l.vy) ? l.vx > 0 : l.vy > 0;  // sort CCW
        if (f1 != f2) return f1 > f2;
        ld ccw = vy * l.vx - vx * l.vy;  // ccw == 0 : parallel
        return z(ccw) ? c * hypot(l.vx, l.vy) < l.c * hypot(vx, vy) : ccw > 0;  // sort by distance
    }
    ld operator / (const Line& l) const { return vy * l.vx - vx * l.vy; }  //cross
};
ld cross(const Pos& d1, const Pos& d2, const Pos& d3) {
    return (d2.x - d1.x) * (d3.y - d2.y) - (d2.y - d1.y) * (d3.x - d2.x);
}
ld A(std::vector<Pos>& H) {
    Pos O = { 0, 0 };
    int l = H.size();
    ld a = 0;
    for (int i = 0; i < l; i++) {
        a += cross(O, H[i], H[(i + 1) % l]);
    }
    return a / 2;
}
Pos IP(const Line& l1, const Line& l2) {
    ld det = l1 / l2;	//ld det = l1.vy * l2.vx - l1.vx * l2.vy;
    return { (l1.c * l2.vx - l1.vx * l2.c) / det, (l1.vy * l2.c - l1.c * l2.vy) / det };
}
bool CW(const Line& l1, const Line& l2, const Line& l) {
    if (l1 / l2 <= 0) return 0;
    Pos p = IP(l1, l2);
    return l.vy * p.x + l.vx * p.y >= l.c;
}
bool HPI(std::vector<Line>& HP, std::vector<Pos>& INX) {
    std::deque<Line> D;
    //std::sort(HP.begin(), HP.end());
    for (const Line& l : HP) {
        if (!D.empty() && z(D.back() / l)) continue;
        while (D.size() >= 2 && CW(D[D.size() - 2], D.back(), l)) D.pop_back();
        while (D.size() >= 2 && CW(l, D[0], D[1])) D.pop_front();
        D.push_back(l);
    }
    while (D.size() > 2 && CW(D[D.size() - 2], D.back(), D[0])) D.pop_back();
    while (D.size() > 2 && CW(D.back(), D[0], D[1])) D.pop_front();
    //if (D.size() < 3) return 0;
    std::vector<Pos> h;
    for (int i = 0; i < D.size(); i++) {
        Line cur = D[i], nxt = D[(i + 1) % D.size()];
        if (cur / nxt <= TOL) return 0;
        h.push_back(IP(cur, nxt));
    }
    INX = h;
    return 1;
}
bool AREA(Pos p[], int m, int N) {  //area
    std::vector<Line> HP;
    for (int i = 0; i < N; i++) {
        ld dy = p[(i + m) % N].y - p[i].y;
        ld dx = p[i].x - p[(i + m) % N].x;  // -(p[i + 1].x - p[i].x)
        ld c = dy * p[i].x + dx * p[i].y;// -hypot(dy, dx) * m;
        HP.push_back({ dy, dx, c });
    }
    std::vector<Pos> INX;
    return HPI(HP, INX);
    //if (!HPI(HP, INX)) return 0;
    //ld area = A(INX);
    //return !z(area);
}
int bi_search(Pos p[], int N) {
    int s = 1, e = N - 1, m;
    while (s < e) {
        m = s + e >> 1;
        if (AREA(p, m, N)) s = m + 1;
        else e = m;
    }
    return s - 1;
}



int main() {
	std::cin.tie(0)->sync_with_stdio(0);
	std::cout.tie(0);
	std::cin >> N;
	for (int i = N - 1; i >= 0; i--) { std::cin >> pos[i].x >> pos[i].y; }
	if (N <= 4) {
		std::cout << "1\n";
		return 0;
	}
	std::cout << bi_search(pos, N) << "\n";
	return 0;
}
{% endhighlight %}
