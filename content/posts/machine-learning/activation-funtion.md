
---
title: "Why Are Activation Functions Needed?"
date: 2026-09-17
draft: false
---

## Why Are Activation Functions Needed?

>When I first learned about neural networks, I wondered If we can simply keep adding more layers, why do we even need activation functions? 
The interesting part is that **without activation functions, adding more layers doesn't actually make the network more powerful.**

Let's see why.

## The short answer

A neural network layer basically performs:

```text
z = Wx + b
````

If we stack many such layers without an activation function, they can be mathematically combined into a single layer.

So a 100-layer network without activation functions can effectively behave like **one linear layer**.

That's why activation functions are important.

---

## What happens without an activation function?

Consider a network with two layers.

The first layer does:

```text
y = W₁x + b₁
```

The second layer takes that output:

```text
z = W₂y + b₂
```

Now substitute `y`:

```text
z = W₂(W₁x + b₁) + b₂
```

Expanding it:

```text
z = W₂W₁x + W₂b₁ + b₂
```

We can rename the terms:

```text
W' = W₂W₁
b' = W₂b₁ + b₂
```

So we get:

```text
z = W'x + b'
```

And that's just another linear layer.

### This is the important part

Even though we started with **two layers**, mathematically they can be collapsed into one.

The same idea applies to 3, 10, or 100 layers.

```text
Layer 1 → Layer 2 → Layer 3 → ... → Layer 100

                 ↓

            One linear transformation
```

So simply adding more linear layers doesn't give the network the ability to learn more complex nonlinear patterns.

---

## Why is linearity a problem?

Real-world relationships are often nonlinear.

For example, imagine predicting whether Alice will pass an exam based on her study hours.

A simple linear relationship might look like:

```text
prediction = w₁ × study_hours + b
```

But real life isn't necessarily like that.

Going from:

```text
1 hour → 2 hours → 3 hours → 4 hours → ... → 20 hours
```

doesn't mean the probability of passing keeps increasing at exactly the same rate.

There can be saturation and other nonlinear effects.

The same problem appears in images.

A neural network might need to learn something like:

```text
Pixels
  ↓
Edges
  ↓
Shapes
  ↓
Parts of objects
  ↓
Complete object
```

These relationships are not simply one big linear transformation.

---

## So what does an activation function do?

An activation function is applied **between layers**.

Instead of:

```text
Layer 1 → Layer 2
```

we have:

```text
Layer 1 → Activation → Layer 2
```

Mathematically:

```text
z = f(Wx + b)
```

For example, with ReLU:

```text
ReLU(x) = max(0, x)
```

So:

```text
x     ReLU(x)

-3       0
-1       0
 0       0
 2       2
 5       5
```

ReLU introduces a nonlinear transformation.

Now our two-layer network becomes:

```text
Layer 1:
h = ReLU(W₁x + b₁)

Layer 2:
z = W₂h + b₂
```

The ReLU in the middle means we can no longer simply combine everything into one linear transformation.

That's the key.

---

## Why does this make deep networks useful?

Now each layer can create a **new representation** of the data.

For example:

```text
Input
  ↓
Layer 1
  ↓
Activation
  ↓
Layer 2
  ↓
Activation
  ↓
Layer 3
  ↓
Output
```

A layer can learn one useful representation, and the next layer can build on it.

For an image, this could look roughly like:

```text
Pixels
  ↓
Edges
  ↓
Simple shapes
  ↓
Complex shapes
  ↓
Object
```

The activation functions allow these transformations to be nonlinear.

This is one of the reasons deep neural networks can model complicated patterns that a simple linear model cannot.

---

## A simple way to visualize it

Imagine you have two groups of points that cannot be separated using one straight line.

A linear model is limited to something like:

```text
      ● ●
    ● ●
-------------  ← one straight boundary
       ○ ○
     ○ ○
```

A neural network with nonlinear activations can transform the representation of the data through multiple layers.

```text
Original data
     ↓
Transform
     ↓
Transform again
     ↓
Better representation
     ↓
Separate the classes
```

So the network isn't simply trying to draw one straight line.

It is **transforming the data step by step**.

---

## One important takeaway

The purpose of an activation function isn't simply to "make the network deeper."

It introduces **nonlinearity**, which allows multiple layers to learn complex relationships.

Without it:

```text
Linear → Linear → Linear
```

is still effectively:

```text
Linear
```

With it:

```text
Linear → Nonlinear → Linear → Nonlinear → ...
```

the network can represent much more complicated functions.

---

## In one sentence

> **Activation functions introduce nonlinearity between neural network layers, allowing deep networks to learn complex patterns that a purely linear model cannot represent.**

And that's the main reason we need them.

```

>If you want a neat visual for your blog: imagine trying to separate two intertwined spirals with a ruler (one straight cut) vs. with many small folds — each layer with activation adds one more "fold."