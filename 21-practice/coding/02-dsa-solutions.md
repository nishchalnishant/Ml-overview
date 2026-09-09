---
module: Practice
topic: Coding Drills
subtopic: DSA Solutions
status: unread
tags: [practice, coding, dsa, solutions]
---
# DSA Screen Problems — Solutions

Problems: [02-dsa-problems.md](02-dsa-problems.md).

Format: the insight, the code, the complexity, and the follow-up the interviewer asks next.

---

## Tier 1 — Hashing and arrays

### 1.1 Two sum

```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []

def all_pairs(nums, target):
    nums = sorted(nums)
    lo, hi, out = 0, len(nums) - 1, []
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target:
            out.append((nums[lo], nums[hi]))
            lo += 1; hi -= 1
            while lo < hi and nums[lo] == nums[lo - 1]: lo += 1
            while lo < hi and nums[hi] == nums[hi + 1]: hi -= 1
        elif s < target: lo += 1
        else: hi -= 1
    return out
```

O(n) time, O(n) space. **Insight:** the hash map turns "does the complement exist" from a
scan into a lookup. **Follow-up:** the all-pairs variant is a different problem — sort plus
two pointers, O(n log n), and the duplicate-skipping loops are what is being graded.

### 1.2 Longest substring without repeats

```python
def length_of_longest_substring(s):
    last, start, best = {}, 0, 0
    for i, c in enumerate(s):
        if c in last and last[c] >= start:
            start = last[c] + 1          # the >= start guard is essential
        last[c] = i
        best = max(best, i - start + 1)
    return best
```

O(n) time, O(min(n, alphabet)) space. **Insight:** the window's left edge never moves
backward. **Trap:** without `last[c] >= start`, a stale index from before the window pulls
`start` backward and the answer is wrong on inputs like `"abba"`.

### 1.3 Group anagrams

```python
from collections import defaultdict

def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        key = tuple(sorted(w))            # or a 26-length count tuple
        groups[key].append(w)
    return list(groups.values())
```

O(n·k log k) with sorted keys; O(n·k) with count-tuple keys, where k is word length.
**Follow-up:** "can you avoid the sort?" — yes, a 26-element count tuple. Mention the tradeoff
rather than waiting to be asked.

### 1.4 Longest consecutive sequence

```python
def longest_consecutive(nums):
    s = set(nums)
    best = 0
    for x in s:
        if x - 1 in s:
            continue                      # only start counting from run beginnings
        y = x
        while y + 1 in s:
            y += 1
        best = max(best, y - x + 1)
    return best
```

O(n) time, O(n) space. **Insight:** the `x - 1 in s` guard makes the inner loop run once per
element *in total* across the whole outer loop, not per outer iteration — that is why this is
O(n) and not O(n²). Be prepared to justify that; it is the entire point of the problem.

### 1.5 Majority element

```python
def majority(nums):
    count, cand = 0, None
    for x in nums:
        if count == 0: cand = x
        count += 1 if x == cand else -1
    return cand
```

O(n) time, O(1) space. Boyer–Moore voting. **Insight:** pair off each majority element with a
non-majority one; since the majority strictly exceeds n/2, at least one survives. **Follow-up:**
"what if no majority is guaranteed?" — add a verification pass.

### 1.6 Product except self

```python
def product_except_self(nums):
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):
        out[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        out[i] *= suffix
        suffix *= nums[i]
    return out
```

O(n) time, O(1) extra space (output excluded). **Insight:** each position's answer is
prefix-product times suffix-product; two passes compute both without division, so zeros need
no special casing.

---

## Tier 2 — Two pointers and sliding window

### 2.1 Remove duplicates in place

```python
def remove_duplicates(nums):
    if not nums: return 0
    w = 1
    for r in range(1, len(nums)):
        if nums[r] != nums[w - 1]:
            nums[w] = nums[r]; w += 1
    return w
```

O(n) time, O(1) space. **Insight:** read and write pointers move at different rates — the
general in-place-compaction pattern, worth naming as such.

### 2.2 Container with most water

```python
def max_area(h):
    lo, hi, best = 0, len(h) - 1, 0
    while lo < hi:
        best = max(best, (hi - lo) * min(h[lo], h[hi]))
        if h[lo] < h[hi]: lo += 1
        else: hi -= 1
    return best
```

O(n) time. **Insight and the thing that is actually graded:** moving the *taller* pointer can
never improve the area, because width shrinks and height is capped by the shorter line.
Therefore moving the shorter one is safe. Candidates who write this without being able to
prove it get less credit than those who can.

### 2.3 Minimum window substring

```python
from collections import Counter

def min_window(s, t):
    if not t or not s: return ""
    need = Counter(t)
    missing = len(t)
    best, lo = (float("inf"), 0, 0), 0
    for hi, c in enumerate(s):
        if need[c] > 0: missing -= 1
        need[c] -= 1
        while missing == 0:
            if hi - lo + 1 < best[0]: best = (hi - lo + 1, lo, hi)
            need[s[lo]] += 1
            if need[s[lo]] > 0: missing += 1
            lo += 1
    return "" if best[0] == float("inf") else s[best[1]:best[2] + 1]
```

O(n) time. **Insight:** `missing` as a single integer replaces comparing two dictionaries on
every step, which is what turns an O(n·k) solution into O(n). Letting counts go negative for
surplus characters is the trick that makes the shrink step correct.

### 2.4 At most k distinct characters

```python
def longest_k_distinct(s, k):
    count, lo, best = defaultdict(int), 0, 0
    for hi, c in enumerate(s):
        count[c] += 1
        while len(count) > k:
            count[s[lo]] -= 1
            if count[s[lo]] == 0: del count[s[lo]]
            lo += 1
        best = max(best, hi - lo + 1)
    return best
```

O(n) time, O(k) space. **Trap:** deleting the key when its count hits zero — without that,
`len(count)` never shrinks and the window never contracts.

### 2.5 Minimum size subarray sum

```python
def min_subarray_len(target, nums):
    lo, total, best = 0, 0, float("inf")
    for hi, x in enumerate(nums):
        total += x
        while total >= target:
            best = min(best, hi - lo + 1)
            total -= nums[lo]; lo += 1
    return 0 if best == float("inf") else best
```

O(n) time. **Follow-up:** "what if negatives are allowed?" — the sliding window breaks
entirely, because the sum is no longer monotone in window size; you need prefix sums plus a
monotonic deque or a sorted structure. Recognizing that the positivity constraint is what
makes the window valid is the mark of understanding rather than pattern-matching.

### 2.6 Three sum

```python
def three_sum(nums):
    nums.sort()
    out = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]: continue
        if nums[i] > 0: break                       # sorted: no zero sum possible
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0: lo += 1
            elif s > 0: hi -= 1
            else:
                out.append([nums[i], nums[lo], nums[hi]])
                lo += 1; hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]: lo += 1
    return out
```

O(n²) time. **Insight:** sorting reduces three-sum to n instances of two-sum-on-sorted.
**What is graded:** duplicate skipping at both the outer index and the inner pointers. Using
a set to dedupe afterward works but is the weaker answer, and interviewers say so.

---

## Tier 3 — Heaps, sorting, intervals

### 3.1 K largest

```python
import heapq

def k_largest(nums, k):
    return heapq.nlargest(k, nums)                     # O(n log k)

class StreamTopK:
    def __init__(self, k):
        self.k, self.heap = k, []
    def add(self, x):
        heapq.heappush(self.heap, x)                   # MIN-heap of size k
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)                   # evict the smallest
```

O(n log k) time, O(k) space. **The insight the streaming follow-up tests:** you keep a
*min*-heap to track the *largest* k, because the cheapest thing to evict is the smallest of
the survivors. Reaching for a max-heap is the intuitive and wrong move.

### 3.2 Merge k sorted lists

```python
def merge_k(lists):
    h = [(l[0], i, 0) for i, l in enumerate(lists) if l]
    heapq.heapify(h)
    out = []
    while h:
        val, li, idx = heapq.heappop(h)
        out.append(val)
        if idx + 1 < len(lists[li]):
            heapq.heappush(h, (lists[li][idx + 1], li, idx + 1))
    return out
```

O(N log k) for N total elements. **Trap:** the list index in the tuple is not decoration — it
breaks ties so the heap never tries to compare two list objects, which raises `TypeError`.

### 3.3 Merge intervals

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    out = []
    for s, e in intervals:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)      # max: the next may be nested
        else:
            out.append([s, e])
    return out
```

O(n log n) time. **Trap:** `max(out[-1][1], e)` rather than `e` — an interval fully contained
in the previous one would otherwise shrink the merged range.

### 3.4 Insert interval

```python
def insert(intervals, new):
    out, i, n = [], 0, len(intervals)
    while i < n and intervals[i][1] < new[0]:
        out.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= new[1]:
        new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]
        i += 1
    out.append(new)
    return out + intervals[i:]
```

O(n) — no sort needed, since the input is already sorted. **Insight:** three phases —
strictly before, overlapping, strictly after. Naming them before coding makes this
straightforward instead of a mess of conditionals.

### 3.5 Meeting rooms II

```python
def min_rooms(intervals):
    starts = sorted(s for s, _ in intervals)
    ends = sorted(e for _, e in intervals)
    rooms = best = j = 0
    for s in starts:
        while j < len(ends) and ends[j] <= s:
            rooms -= 1; j += 1
        rooms += 1
        best = max(best, rooms)
    return best
```

O(n log n) time, O(n) space. **Insight:** you never need to know *which* meeting ended, only
that one did — so the start and end times can be decoupled and swept independently. The heap
solution is equally valid; this one is cleaner. **Generalization worth stating:** this is the
maximum-overlap sweep-line, the same shape as peak concurrent connections.

### 3.6 Median of a stream

```python
class MedianFinder:
    def __init__(self):
        self.lo, self.hi = [], []          # lo: max-heap (negated); hi: min-heap
    def add(self, x):
        heapq.heappush(self.lo, -heapq.heappushpop(self.hi, x))
        if len(self.lo) > len(self.hi):
            heapq.heappush(self.hi, -heapq.heappop(self.lo))
    def get_median(self):
        if not self.hi: raise ValueError("empty")
        if len(self.hi) > len(self.lo): return float(self.hi[0])
        return (self.hi[0] - self.lo[0]) / 2
```

O(log n) add, O(1) query. **Insight:** two heaps split the data at the median; the
`pushpop`-then-push idiom keeps them balanced and correctly ordered in one move. Python has
no max-heap, so negation is the standard workaround — say that out loud rather than letting
the interviewer wonder if the minus signs are a bug.

### 3.7 Top-k frequent

```python
def top_k_frequent(nums, k):
    return [x for x, _ in Counter(nums).most_common(k)]      # O(n log k)

def top_k_bucket(nums, k):
    counts = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for x, c in counts.items():
        buckets[c].append(x)
    out = []
    for c in range(len(buckets) - 1, 0, -1):
        out.extend(buckets[c])
        if len(out) >= k: return out[:k]
    return out
```

**Insight for the O(n) version:** frequency is bounded by n, so it can index an array —
bucket sort applies whenever the sort key is a small bounded integer. That observation is the
transferable part.

---

## Tier 4 — Graphs, trees, DP

### 4.1 Number of islands

```python
def num_islands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    seen, count = set(), 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] != "1" or (r, c) in seen: continue
            count += 1
            stack = [(r, c)]                     # iterative: no recursion limit
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y+1,x), (y-1,x), (y,x+1), (y,x-1)):
                    if 0 <= ny < R and 0 <= nx < C and grid[ny][nx] == "1" \
                       and (ny, nx) not in seen:
                        seen.add((ny, nx)); stack.append((ny, nx))
    return count
```

O(R·C) time and space. **Why iterative:** a 1000×1000 all-land grid blows Python's recursion
limit. Mentioning that unprompted is a production-thinking signal. **Marking seen at push
time, not pop time,** prevents the same cell entering the stack many times.

### 4.2 Course schedule

```python
from collections import deque

def course_order(n, prereqs):
    adj = [[] for _ in range(n)]
    indeg = [0] * n
    for course, pre in prereqs:
        adj[pre].append(course); indeg[course] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    return order if len(order) == n else []      # short order == cycle
```

O(V + E) time. **Insight:** Kahn's algorithm answers both questions at once — a valid ordering
if one exists, and cycle detection for free from `len(order) != n`. Say that; it is the reason
this formulation is preferred over DFS with colors.

### 4.3 Level-order traversal

```python
def level_order(root):
    if not root: return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):                  # snapshot the level size FIRST
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        out.append(level)
    return out
```

O(n) time. **Trap:** capturing `len(q)` before the inner loop. Iterating over a queue you are
appending to inside the loop merges every level into one.

### 4.4 Lowest common ancestor

```python
def lca(root, p, q):
    if root is None or root is p or root is q:
        return root
    left, right = lca(root.left, p, q), lca(root.right, p, q)
    if left and right: return root               # split point == LCA
    return left or right
```

O(n) time. **Insight:** the recursion returns "a target, or the LCA if already found." When
both sides return non-null, the current node is where the two paths diverge. Being able to
state that invariant is worth more than the four lines of code.

### 4.5 Coin change

```python
def coin_change(coins, amount):
    dp = [0] + [float("inf")] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return -1 if dp[amount] == float("inf") else dp[amount]
```

O(amount · len(coins)) time, O(amount) space. **The recurrence, stated first:**
`dp[a] = 1 + min(dp[a - c])` over every coin `c ≤ a`. **Why greedy fails:** with
`coins = [1, 3, 4]` and `amount = 6`, greedy takes 4 + 1 + 1 = three coins; the optimum is
3 + 3 = two. Have that counterexample ready — it is the standard follow-up.

### 4.6 Longest common subsequence

```python
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

O(m·n) time and space, reducible to O(min(m, n)) space by keeping two rows. **Insight:** the
`+1` on a match versus the `max` on a mismatch is the entire recurrence; the 1-indexed table
with a zero row and column removes all boundary conditionals.

### 4.7 Shortest path in an unweighted graph

```python
def shortest_path(adj, src, dst):
    q, dist = deque([src]), {src: 0}
    while q:
        u = q.popleft()
        if u == dst: return dist[u]
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return -1
```

O(V + E). **The answer to "why BFS, not DFS":** BFS expands nodes in nondecreasing distance
order, so the first time a node is reached is via a shortest path. DFS finds *a* path with no
optimality guarantee, and can wander arbitrarily far before reaching a neighbor. Follow-up:
with weighted edges you need Dijkstra, and BFS is exactly Dijkstra when all weights are equal.

### 4.8 Validate BST

```python
def is_valid_bst(node, lo=float("-inf"), hi=float("inf")):
    if node is None: return True
    if not (lo < node.val < hi): return False
    return (is_valid_bst(node.left, lo, node.val)
            and is_valid_bst(node.right, node.val, hi))
```

O(n) time. **Why the obvious solution is wrong:** checking only `left.val < node.val <
right.val` at each node is a *local* check, and the BST property is global. The tree
`5 → (left 1 → (right 6))` passes every local check but is not a BST, because 6 sits in the
left subtree of 5. The bounds must be threaded down the recursion. An in-order traversal
verifying strict increase is the equally valid alternative.
