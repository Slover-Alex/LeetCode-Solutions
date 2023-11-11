import java.util.*;

import javax.swing.tree.TreeNode;

import org.w3c.dom.Node;

import apple.laf.JRSUIUtils.Tree;

public class Solution {

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right, Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return traverser(root, p, q);
    }

    List<List<Integer>> bigList = new ArrayList<>();
    Queue<TreeNode> queue = new LinkedList<>();

    public List<List<Integer>> levelOrder(TreeNode root) {

        if (root != null) {
            traverser2(root);
        }
        return bigList;
    }

    public void traverser2(TreeNode root) {

        if (root == null) {
            return;
        }

        queue.add(root);
        while (!queue.isEmpty()) {
            int nodesInLevel = queue.size();
            ArrayList<Integer> smallList = new ArrayList<>();

            for (int i = 0; i < nodesInLevel; i++) {
                TreeNode current = queue.remove();
                smallList.add(Integer.valueOf(current.val));

                if (i == nodesInLevel - 1) {
                    bigList.add(smallList);
                }

                if (current.left != null) {
                    queue.add(current.left);
                }

                if (current.right != null) {
                    queue.add(current.right);
                }
            }
        }
        return;
    }

    List<List<Integer>> bigList3 = new ArrayList<>();
    Deque<TreeNode> deque = new LinkedList<>();

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        if (root != null) {
            traverser3(root);
        }
        return bigList3;
    }

    public void traverser3(TreeNode root) {

        if (root == null) {
            return;
        }
        TreeNode current;
        deque.add(root);
        while (!queue3.isEmpty()) {
            int nodesInLevel = deque.size();
            ArrayList<Integer> smallList3 = new ArrayList<>();
            int zigzag = 1;

            for (int i = 0; i < nodesInLevel; i++) {
                if (zigzag > 0) {
                    current = deque.getFirst();
                } else {
                    current = deque.getLast();
                }

                smallList3.add(Integer.valueOf(current.val));

                if (i == nodesInLevel - 1) {
                    bigList3.add(smallList3);
                    zigzag = zigzag * -1;
                }

                if (zigzag > 0) {
                    if (current.left != null) {
                        deque.add(current.left);
                    }

                    if (current.right != null) {
                        deque.add(current.right);
                    }

                } else {
                    if (current.right != null) {
                        deque.addLast(current.right);
                    }
                    if (current.left != null) {
                        deque.addLast(current.left);
                    }
                }

            }
        }
        return;
    }

    // Given the root of the binary tree, return the values of the right most node
    // at each level of the tree from top to bottom.
    // Use BFS and traverse from right to left and add the right most value until
    // you get to the next level 

    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> list = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();

        q.add(root);

        while (!q.isEmpty()) {
            int NodesInLevel = q.size();
            Double levelSum = 0.0;
            for (int i = 0; i < NodesInLevel; i++) {
                TreeNode current = q.poll();
                levelSum += Integer.valueOf(current.val);

                if (i == NodesInLevel - 1) {
                    list.add(levelSum / NodesInLevel);
                }

                if (current.left != null) {
                    q.add(current.left);
                }

                if (current.right != null) {
                    q.add(current.right);
                }
            }
        }
        return list;
    }

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);

        if (root == null) {
            return list;
        }

        while (!q.isEmpty()) {
            int level = q.size();
            for (int i = 0; i < level; i++) {
                TreeNode current = q.poll();

                if (i == level - 1) {
                    list.add(Integer.valueOf(current.val));
                }

                if (current.left != null) {
                    q.add(current.left);
                }

                if (current.right != null) {
                    q.add(current.right);
                }
            }
        }

        return list;
    }

    TreeNode ancestor = null;

    public TreeNode traverser(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return null;
        }

        // If found p or q, propogate root upwards
        if (root == p || root == q) {
            return root;
        }

        // Otherwise traverse on left and right children
        TreeNode left = traverser(root.left, p, q);
        TreeNode right = traverser(root.right, p, q);

        // If the return of left and right from the upward propogated values are
        // non-null, return the root
        if (left != null && right != null) {
            return root;
        }

        // If p or q is only found in the left subtree, return that result (must contain
        // both p and q)
        else if (left != null) {
            return left;
        }

        // If either p or q is found in the right subtree, return that result (must
        // contain both p and q)
        else {
            return right;
        }

    }

    class BSTIterator {

        Queue<Integer> q = new LinkedList<>();

        public void traverser(TreeNode root) {
            if (root == null) {
                return;
            }

            traverser(root.left);
            q.add(Integer.valueOf(root.val));
            traverser(root.right);

            return;
        }

        public BSTIterator(TreeNode root) {
            traverser(root);
        }

        public int next() {
            return q.remove().intValue();
        }

        public boolean hasNext() {
            return (q.peek() != null);
        }
    }

    public boolean hasPathSum(TreeNode root, int targetSum) {
        // inOrder traversal of a tree for DFS?

        if (root == null) {
            return false;
        } else {
            targetSum -= root.val;
            pathSumHelper(root, targetSum);
            return bool;
        }

    }

    public boolean bool = false;

    public void pathSumHelper(TreeNode root, int targetSum) {

        targetSum -= root.val;

        if (root.left != null) {
            pathSumHelper(root.left, targetSum);
        }

        if (root.left == null && root.right == null && targetSum == 0) {
            bool = true;
            return;
        }

        else if (root.left == null && root.right == null && targetSum != 0) {
            targetSum += root.val;
        }

        if (root.right != null) {
            pathSumHelper(root.right, targetSum);
        }

    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        HashMap<Integer, Integer> map = new HashMap<>();
        Integer index = 0;
        for (int num : inorder) {
            map.put(Integer.valueOf(num), index);
            index++;
        }

        Stack<TreeNode> q = new Stack<>();

        TreeNode root = new TreeNode(preorder[0]);
        TreeNode rightMost = root;
        TreeNode previous = root;
        Integer current;
        TreeNode ans = root; // pointer to the root that we will return
        q.add(root);

        for (int i = 1; i < preorder.length; i++) {
            current = map.get(Integer.valueOf(preorder[i]));

            if (current > map.get(Integer.valueOf(rightMost.val))) {
                rightMost.right = new TreeNode(preorder[i], null, null);
                q.push(rightMost.right);
                // previous = previous.left;
            }

            while (current < map.get(Integer.valueOf(previous.val)) && q.peek() != null) {
                previous = map.get(Integer.valueOf(q.peek().val));

                if (current > map.get(Integer.valueOf(previous.val))) {
                    previous.right = new TreeNode(preorder[i], null, null);
                    q.push(previous.right);
                    break;
                }

                if (q.peek() == null) {
                    previous.left = new TreeNode(preorder[i], null, null);
                    q.push(previous.left);
                }

            }
        }

        return ans;

    }

    public int maxVowels(String s, int k) {
        String[] arr = s.split("");
        TreeMap<String, Boolean> map = new TreeMap<>();
        map.put("a", true);
        map.put("i", true);
        map.put("e", true);
        map.put("o", true);
        map.put("u", true);

        int currCount = 0;
        int maxCount = 0;

        for (int i = 0; i < arr.length; i++) {
            if (map.containsKey(arr[i])) {
                currCount += 1;
            }

            if (i >= k - 1) {
                if (currCount > maxCount) {
                    maxCount = currCount;
                }
                if (map.get(arr[i - k + 1]) == true) {
                    currCount -= 1;
                }
            }
        }

        return maxCount;
    }

    public int largestAltitude(int[] gain) {
        int max = 0;
        int current = 0;

        for (int num : gain) {
            current += num;
            if (current > max) {
                max = current;
            }
        }
        return max;
    }

    public double findMaxAverage(int[] nums, int k) {
        int currentSum = 0;
        double maxAVG = Integer.MIN_VALUE;

        for (int i = 0; i < nums.length; i++) {
            // If we're currently at index which creates k size window
            currentSum += nums[i];
            if (i >= k - 1) {
                double currentAvg = (double) currentSum / k;
                if (currentAvg > maxAVG) {
                    maxAVG = currentAvg;
                }
                // remove the first element in the window
                currentSum -= nums[i - k + 1];
            }

        }
        return maxAVG;
    }

    public int maxOperations(int[] nums, int k) {

        HashMap<Integer, Integer> map = new HashMap<>();

        int operations = 0;

        for (int i = 0; i < nums.length; i++) {

            // to check if that k-nums[i] present and had some value left or already paired
            if (map.containsKey(k - nums[i]) && map.get(k - nums[i]) > 0) {
                operations++;
                map.put(k - nums[i], map.get(k - nums[i]) - 1);
            }

            else {
                // getOrDefault is easy way it directly checks if value is 0 returns 0 where I
                // added 1
                // and if some value is present then it returns that value "similar to
                // map.get(i)" and adds 1 to it
                map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
            }
        }
        return operations;
    }

    static int maxVolume;

    public void calcVolume(int x, int y, int[] heights) {
        int trueHeight = Math.min(heights[x], heights[y]);
        int width = Math.abs(x - y);
        int currentVolume = trueHeight * width;
        if ((currentVolume) > maxVolume) {
            maxVolume = currentVolume;
        }
    }

    public int maxArea(int[] height) {
        maxVolume = 0;
        int x = 0;
        int y = height.length - 1;

        while (x != y && x < y) {

            calcVolume(x, y, height);
            if (height[x] < height[y]) {
                x += 1;
            } else if (height[x] > height[y]) {
                y -= 1;
            } else if (height[x] == height[y]) {
                y -= 1;
                x += 1;
            }
        }
        return maxVolume;
    }

    static int count = 1;

    public int pivotIndex(int[] nums) {
        int total = 0;
        int currentSum = 0;

        for (int n : nums) {
            total += n;
        }

        for (int i = 0; i < nums.length; i++) {

            if (total - currentSum - nums[i] == currentSum) {
                return i;
            }

            currentSum += nums[i];
        }
        return -1;
    }

    // propogate upwards the number of nodes greater or equal to current sum
    // since each node has left and right children, maybe return the sum of left and
    // right good nodes
    // this means for each current node, recursively traverse the tree

    int sum = 0;
    TreeNode prev = null;

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }

        // Post-order traversal: left, right, root
        flatten(root.right);
        flatten(root.left);

        // Flatten the tree
        root.left = null;
        root.right = prev;

        // Update the previous node to current root
        prev = root;
    }

    public TreeNode temporary;

    public TreeNode flipper(TreeNode root) {
        temporary = root.left;
        root.left = root.right;
        root.right = temporary;

        if (root.left != null) {
            invertTree(root.left);
        }
        if (root.right != null) {
            invertTree(root.right);
        }

        return root;
    }

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        return flipper(root);
    }

    public Node connect(Node root) {
        // check for null input
        if (root == null)
            return root;
        // make a queue for bfs
        Queue<Node> queue = new ArrayDeque<>();
        queue.add(root);
        // going through the nodes in the queue

        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Node curr = queue.poll();

                // If the node is NOT the last node in its level:
                if (i < size - 1)
                    curr.next = queue.peek();

                if (curr.left != null)
                    queue.add(curr.left);
                if (curr.right != null)
                    queue.add(curr.right);
            }
        }
        return root;
    }

    public boolean isSameTree(TreeNode p, TreeNode q) {
        ArrayList<Integer> list1 = new ArrayList<>();
        ArrayList<Integer> list2 = new ArrayList<>();
        TreeToList(p, list1);
        TreeToList(q, list2);
        return (list1.equals(list2));
    }

    public ArrayList TreeToList(TreeNode root, ArrayList<Integer> list) {
        if (root == null) {
            list.add(null);
            return list;
        }

        else {
            list.add(Integer.valueOf(root.val));
        }

        TreeToList(root.left, list);
        TreeToList(root.right, list);

        return list;
    }

    public int countNode(TreeNode root, int currentMax) {

        if (root.val >= currentMax) {
            sum += 1;
            currentMax = Math.max(root.val, currentMax);
        }

        if (root.left != null) {
            sum += countNode(root.left, currentMax);
        }

        if (root.right != null) {
            sum += countNode(root.right, currentMax);
        }

        return sum;
    }

    public int goodNodes(TreeNode root) {
        int L = 0;
        int R = 0;
        if (root.left != null) {
            L = countNode(root.left, root.val);
        }
        if (root.right != null) {
            R = countNode(root.right, root.val);
        }
        return L + R;
    }

    Queue<Integer> q1 = new LinkedList<>();
    Queue<Integer> q2 = new LinkedList<>();

    public boolean fin;

    public void addLeaves(TreeNode root, Queue<Integer> queue) {

        if (root.left != null) {
            addLeaves(root.left, queue);
        }

        // if at leaf node -> add it to list
        if (root.right == null && root.left == null) {
            queue.add(Integer.valueOf(root.val));
        }

        if (root.right != null) {
            addLeaves(root.right, queue);
        }

    }

    public boolean leafSimilar(TreeNode root1, TreeNode root2) {
        addLeaves(root1, q1);
        addLeaves(root2, q2);

        fin = true;

        while (q1.peek() != null || q2.peek() != null) {

            if (q1.poll() != q2.poll()) {
                fin = false;
                break;
            }
        }

        return fin;

    }

    public int maxDepth(TreeNode root) {

        // not even a node
        if (root == null) {
            return 0;
        }

        // at a leaf node
        if (root.left == null && root.right == null) {
            return 1;
        }

        else {
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        }
    }

    public ListNode oddEvenList(ListNode head) {

        boolean currIsOdd = false;

        if (head == null || head.next == null) {
            return null;
        }

        ListNode odd = new ListNode(head.val, null);
        ListNode oddFirst = odd;

        ListNode even = new ListNode(head.next.val, null);
        ListNode evenFirst = even;

        // currenly pointing at second node!
        head = head.next;

        while (head != null) {

            if (currIsOdd == true) {
                odd.next = new ListNode(head.val, null);
                odd = odd.next;
                currIsOdd = false;
            }

            else {
                even.next = new ListNode(head.val, null);
                even = even.next;
                currIsOdd = true;
            }

            head = head.next;
        }

        odd.next = evenFirst.next;

        return oddFirst;
    }

    public ListNode deleteMiddle(ListNode head) {

        ListNode ptr = head;
        ListNode first = head;

        if (head == null || head.next == null) {
            return null;
        }

        int count = 0;
        while (head != null) {
            head = head.next;
            count++;
        }

        Double mid = Math.floor(count / 2);

        for (int i = 0; i < count; i++) {

            // if next node is middle one, skip it!
            if (i + 1 == mid) {
                ptr.next = ptr.next.next;
                break;
            }

            ptr = ptr.next;
        }

        return first;

    }
}
