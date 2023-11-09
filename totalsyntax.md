---
title: 사용법
date: 2023-10-30
link: http://www.google.com/
categories:
- Markdown
tags:
- Markdown
photos: # 사진 걸기
description: 세부묘사
---

주석으로 끊으면 뒤가 안 나오나?
<!-- more -->

미리 보기
bundle exec jekyll serve


진짜?

# 헤딩

## 헤딩2

[링크]()

**굵게**

<u>밑줄</u>

~~취소선~~

`블록`

> 들여쓰기

<dl><dt>아래쪽 타이틀</dt><dd>이건 뭐지?</dd></dl>

1. 순서
2. 있는
3. 리스트

- 순서
- 없는
- 리스트

| 테 | 이 | 블 |
| --- | --- | --- |
| 1 | 2 | 3 |
| 1 | 2 | 3 |

<sup>위 첨자</sup>
<sub>아래 첨자</sub>
<cite>비껴쓰기</cite>

{% highlight cpp %}
int X = 1;  // 번호 없는 코드 블록
{% endhighlight %}

{% highlight cpp linenos %}
bool OK(int x) {
    return x > 0;  // 번호 있는 코드 블록
}
{% endhighlight %}

이건 쓸 일이 있을까?

{% gist 996818 %}

![사진](경로)

See emoji cheat sheet for more detail :wink: : <https://www.webpagefx.com/tools/emoji-cheat-sheet/>.