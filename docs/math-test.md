---
layout: page
title: Math Test
---

# Math Expression Test

This page tests various math expressions to verify that kramdown and MathJax are working correctly.

## Inline Math

Inline math expressions like $E = mc^2$ should render properly.

Another example: $\frac{1}{2} + \frac{1}{3} = \frac{5}{6}$

## Block Math

Display math should work with double dollar signs:

$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

Or with LaTeX delimiters:

\[
\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}
\]

## Complex Equations

$$
\begin{align}
\nabla \cdot \vec{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \vec{B} &= 0 \\
\nabla \times \vec{E} &= -\frac{\partial \vec{B}}{\partial t} \\
\nabla \times \vec{B} &= \mu_0 \vec{J} + \mu_0 \epsilon_0 \frac{\partial \vec{E}}{\partial t}
\end{align}
$$

## Code Blocks

Regular code blocks should still work:

```python
def hello_world():
    print("Hello, World!")
```

## Mixed Content

You can mix math with text: The quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

For display math:

$$
f(x) = \int_{0}^{x} \frac{1}{\sqrt{1-t^2}} dt
$$ 