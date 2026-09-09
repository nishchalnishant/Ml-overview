---
module: Practice
topic: Coding Drills
subtopic: DSA Problems
status: unread
tags: [practice, coding, dsa, algorithms]
---
# DSA Screen Problems

24 problems biased toward what actually appears in ML interview loops: arrays,
hashing, heaps, two pointers, intervals, and light graph/DP. Not a general
competitive-programming set.

**Solutions:** [02-dsa-solutions.md](02-dsa-solutions.md) — attempt first.

**Ground rules.** State the brute force and its complexity before optimizing. State
final time and space complexity unprompted. Handle empty input, single element, and
all-duplicates without being asked.

---

## Tier 1 — Hashing and arrays (15 min each)

**1.1** Given an array and a target, return indices of two numbers summing to the
target. Then: return *all* unique pairs, not just one.

**1.2** Find the length of the longest substring without repeating characters.

**1.3** Group a list of strings into anagram groups.

**1.4** Given an unsorted array, find the length of the longest run of consecutive
integers present. O(n) required — sorting is the answer you must beat.

**1.5** Return the majority element (appears more than n/2 times) in O(1) extra space.

**1.6** Given an array of integers, return the product of all elements except self, in
O(n) time, without using division.

---

## Tier 2 — Two pointers and sliding window (20 min each)

**2.1** Given a sorted array, remove duplicates in place and return the new length.

**2.2** Container with most water: given heights, find two lines forming the largest
container.

**2.3** Minimum window substring: smallest window in `s` containing all characters of
`t` (with multiplicity).

**2.4** Longest substring with at most `k` distinct characters.

**2.5** Given an array of positive integers and a target, find the minimal-length
contiguous subarray with sum ≥ target.

**2.6** Three-sum: all unique triplets summing to zero. Duplicate handling is the
whole problem.

---

## Tier 3 — Heaps, sorting, intervals (20 min each)

**3.1** Return the k largest elements in an array. Then: what changes if the array is
a stream you cannot hold in memory?

**3.2** Merge k sorted lists into one sorted list.

**3.3** Merge all overlapping intervals in a list of `[start, end]` pairs.

**3.4** Given intervals and a new interval, insert it and merge as needed.

**3.5** Meeting rooms: given intervals, find the minimum number of rooms required.

**3.6** Find the median of a stream of integers, supporting `add(x)` and
`get_median()` efficiently.

**3.7** Top-k frequent elements in an array. Give both the heap solution and the
O(n) one.

---

## Tier 4 — Graphs, trees, DP (25 min each)

**4.1** Number of connected components ("islands") in a 2D grid of land and water.

**4.2** Course schedule: given prerequisite pairs, determine whether all courses can
be finished. Then: return a valid ordering.

**4.3** Level-order traversal of a binary tree, returning a list per level.

**4.4** Lowest common ancestor of two nodes in a binary tree.

**4.5** Climbing stairs / coin change: minimum coins to make an amount. State the
recurrence before writing code.

**4.6** Longest common subsequence of two strings.

**4.7** Word ladder or shortest path in an unweighted graph — explain why BFS and not
DFS.

**4.8** Given a binary tree, determine if it is a valid binary search tree. The
obvious solution is wrong; find out why.
