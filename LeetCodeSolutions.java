import java.util.*;

public class LeetCodeSolutions {

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

    class Node {
        public int val;
        public List<Node> neighbors;

        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }

        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }

    public int numIslands(char[][] grid) {
        // iterate through each cell on the grid using a for loop
        // when you come across an island, increment islands and use BFS to find the
        // whole island
        // when using DFS or BFS, make sure to never go outside the boundaries of the
        // grid

        if (grid == null || grid.length == 0) {
            return 0;
        }

        int islands = 0;

        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    islands++;
                    dfs(grid, i, j);
                }
            }
        }

        return islands;

    }

    private void dfs(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0') {
            return;
        }

        // mark visited cell as visited
        grid[i][j] = '0';

        // explore neighboring cells
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }

    public void solve(char[][] board) {
        // Given an m x n matrix board containing 'X' and 'O',
        // capture all regions that are 4-directionally surrounded by 'X'
        // use DFS to mark the cells that aren't on the borders (i.e char.length - 1)
        // Use a Queue: only add cells that aren't on the edge

        // An 'O' is surrounded if there is NO path from it to the boundary of the
        // matrix
        // (ie. row index 0, column index 0, row index matrix.length-1, column index
        // matrix[0].length-1)

        HashMap<Character, Character> map = new HashMap<>();

        if (board.length == 0 || board == null) {
            return;
        }

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'O') {
                    dfs2(board, i, j);
                }
            }
        }
    }

    private void dfs2(char[][] board, int i, int j) {
        if (i <= 0 || i >= board.length - 1 || j <= 0 || j >= board[i].length - 1 || board[i][j] == 'X') {
            return;
        }

        board[i][j] = 'X';
        dfs2(board, i, j + 1);
        dfs2(board, i, j - 1);
        dfs2(board, i + 1, j);
        dfs2(board, i - 1, j);
        return;
    }

    public Node cloneGraph(Node node) {

        Node head = new Node(node.val);
        for (int i = 0; i < node.neighbors.size(); i++) {
            Node current = node.neighbors.get(i);
            if (!head.neighbors.contains(current)) {
                head.neighbors.add(current);
            }
        }

        return head;
    }

    public boolean canFinish(int n, int[][] prerequisites) {
        List<Integer>[] adj = new List[n]; /* create an empty a list of neighbors for each node */
        int[] indegree = new int[n]; /* represents the indegree of each node */
        List<Integer> ans = new ArrayList<>(); /* represents the topological sort order which must equal to size n */

        for (int[] pair : prerequisites) {
            int course = pair[0];
            int prerequisite = pair[1];
            if (adj[prerequisite] == null) { /*
                                              * If there are no neighbors for this prerequisite node, create it.
                                              * Otherwise add the course
                                              */
                adj[prerequisite] = new ArrayList<>();
            }
            adj[prerequisite].add(course);
            indegree[course]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; i++) { /* adds to the queue all the nodes that have no dependencies */
            if (indegree[i] == 0) {
                queue.offer(i);
            }
        }

        while (!queue.isEmpty()) {
            int current = queue.poll();
            ans.add(current);

            if (adj[current] != null) { /* If this node has neighbors */
                for (int next : adj[current]) { /* for each neighboring node */
                    indegree[next]--; /* reduce the indegree by 1 */
                    if (indegree[next] == 0) { /* If node has no other dependencies, add it to the queue */
                        queue.offer(next);
                    }
                }
            }
        }

        return ans.size() == n;
    }

    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Map<String, Double>> graph = buildGraph(equations, values);
        double[] results = new double[queries.size()];

        for (int i = 0; i < queries.size(); i++) {
            List<String> query = queries.get(i);
            String dividend = query.get(0);
            String divisor = query.get(1);

            if (!graph.containsKey(dividend) || !graph.containsKey(divisor)) {
                results[i] = -1.0;
            } else {
                results[i] = bfs(dividend, divisor, graph);
            }
        }

        return results;
    }

    private Map<String, Map<String, Double>> buildGraph(List<List<String>> equations, double[] values) {
        Map<String, Map<String, Double>> graph = new HashMap<>();

        for (int i = 0; i < equations.size(); i++) {
            List<String> equation = equations.get(i);
            String dividend = equation.get(0);
            String divisor = equation.get(1);
            double value = values[i];

            graph.putIfAbsent(dividend, new HashMap<>());
            graph.putIfAbsent(divisor, new HashMap<>());
            graph.get(dividend).put(divisor, value);
            graph.get(divisor).put(dividend, 1.0 / value);
        }

        return graph;
    }

    private double bfs(String start, String end, Map<String, Map<String, Double>> graph) {
        Queue<Pair<String, Double>> queue = new LinkedList<>();
        Set<String> visited = new HashSet<>();
        queue.offer(new Pair<>(start, 1.0));

        while (!queue.isEmpty()) {
            Pair<String, Double> pair = queue.poll();
            String node = pair.getKey();
            double value = pair.getValue();

            if (node.equals(end)) {
                return value;
            }

            visited.add(node);

            for (Map.Entry<String, Double> neighbor : graph.get(node).entrySet()) {
                String neighborNode = neighbor.getKey();
                double neighborValue = neighbor.getValue();

                if (!visited.contains(neighborNode)) {
                    queue.offer(new Pair<>(neighborNode, value * neighborValue));
                }
            }
        }

        return -1.0;
    }

    private Map<String, List<String>> createGraph(String startGene, String[] bank) {
        Map<String, List<String>> graph = new HashMap<>();

        for (int i = 0; i < bank.length; i++) {
            graph.putIfAbsent(bank[i], new ArrayList<>());

            // for each subsequent gene
            for (int j = i + 1; j < bank.length; j++) {
                if (isOneMutationAway(bank[i], bank[j])) { // if the i'th gene is one mutation away from i+1'th gene
                    graph.get(bank[i]).add(bank[j]); // add their relationship to the graph
                    graph.putIfAbsent(bank[j], new ArrayList<>());
                    graph.get(bank[j]).add(bank[i]);
                }
            }

            // Check if startGene is one mutation away from any gene in bank
            if (isOneMutationAway(startGene, bank[i])) {
                graph.putIfAbsent(startGene, new ArrayList<>());
                graph.get(startGene).add(bank[i]);
            }
        }
        return graph;
    }

    private boolean isOneMutationAway(String gene1, String gene2) {
        int count = 0;
        for (int i = 0; i < gene1.length(); i++) {
            if (gene1.charAt(i) != gene2.charAt(i)) {
                count++;
            }
            if (count > 1) {
                return false;
            }
        }
        return count == 1;
    }

    public int minMutation(String startGene, String endGene, String[] bank) {
        Map<String, List<String>> graph = createGraph(startGene, bank);

        // If there's no endGene in the bank, return -1
        if (!graph.containsKey(endGene)) {
            return -1;
        }

        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(startGene);
        visited.add(startGene);

        int mutations = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String current = queue.poll();
                if (current.equals(endGene)) {
                    return mutations;
                }
                List<String> neighbors = graph.getOrDefault(current, new ArrayList<>());
                for (String neighbor : neighbors) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.offer(neighbor); // searching each neighbor before moving on
                    }
                }
            }
            mutations++;
        }
        return -1;
    }

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        for (int j = 0, i = m; j < n; i++) {
            nums1[i] = nums2[j];
            j++;
        }
        Arrays.sort(nums1);
    }

    public int removeDuplicates(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        List<Integer> list = new LinkedList<>();
        for (int i = 0; i < nums.length; i++) {
            Integer current = Integer.valueOf(nums[i]);
            map.putIfAbsent(current, 0);
            if (map.get(current) < 2) {
                list.add(current);
                Integer toBeSet = map.get(current);
                toBeSet++;
                map.put(current, toBeSet);
            }
        }

        for (int j = 0; j < list.size(); j++) {
            nums[j] = list.get(j).intValue();
        }

        return list.size();
    }

    public void rotate(int[] nums, int k) {
        int leftover = 0;
        if (k > nums.length) {
            leftover = k % nums.length;
            doIt(nums, leftover);
        } else {
            doIt(nums, k);
        }
        return;
    }

    public void doIt(int[] nums, int k) {

        int[] bers = new int[k];
        int[] num = new int[nums.length - k];

        for (int i = nums.length - k, h = 0; i < nums.length; i++) {
            bers[h] = nums[i];
            h++;
        }
        for (int j = 0; j < nums.length - k; j++) {
            num[j] = nums[j];
        }
        for (int x = 0; x < bers.length; x++) {
            nums[x] = bers[x];
        }
        for (int y = bers.length, g = 0; y < nums.length; y++) {
            nums[y] = num[g];
            g++;
        }

        return;
    }

    public int maxProfit(int[] prices) {
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i] < prices[i + 1]) {
                profit += prices[i + 1] - prices[i];
            }
        }
        return profit;
    }

    public boolean canJump(int[] nums) {
        // find maxJumpToIndex, within that range, check to see if there's one that
        // holds more and use it
        int maxJumpToIndex = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            if (i == maxJumpToIndex && nums[i] == 0) {
                return false;
            } else if (nums[i] > 0 && nums[i] + i > maxJumpToIndex) {
                maxJumpToIndex = nums[i] + i;
            }
        }
        return maxJumpToIndex >= nums.length - 1;
    }

    public int minSubArrayLen(int target, int[] nums) {
        int left = 0;
        int minSize = Integer.MAX_VALUE;
        int runningSum = 0;

        for (int right = 0; right < nums.length; right++) {
            runningSum += nums[right];

            if (runningSum >= target) {
                if (right + 1 - left < minSize) {
                    minSize = right + 1 - left;
                }
                while (runningSum > target) {
                    runningSum -= nums[left];
                    left++;

                    if (runningSum >= target && right + 1 - left < minSize) {
                        minSize = right + 1 - left;
                    }
                }
            }

        }

        if (minSize == Integer.MAX_VALUE) {
            return minSize = 0;
        }

        return minSize;
    }


    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Long> stack1 = new Stack<>();
        Stack<Long> stack2 = new Stack<>();

        long num1 = 0;
        long num2 = 0;
        long pow1 = 0;
        long pow2 = 0;
        long total = 0;

        while (l1 != null) {
            pow1++;
            stack1.add((long) l1.val);
            l1 = l1.next;
        }
        while (l2 != null) {
            pow2++;
            stack2.add((long) l2.val);
            l2 = l2.next;
        }
        while (!stack1.isEmpty()) {
            num1 += stack1.pop() * Math.pow(10, pow1 - 1);
            pow1--;
        }
        while (!stack2.isEmpty()) {
            num2 += stack2.pop() * Math.pow(10, pow2 - 1);
            pow2--;
        }

        total = num1 + num2;
        String[] nums = Long.toString(total).split("");
        ListNode answer = new ListNode((int) (Long.parseLong(nums[nums.length - 1])));

        ListNode head = answer;

        for (int i = 2; i < nums.length + 1; i++) {
            answer.next = new ListNode((int) Long.parseLong(nums[nums.length - i]));
            answer = answer.next;
        }

        return head;

    }

    public Node copyRandomList(Node head) {

        if (head == null) {
            return null;
        }

        Map<Node, Node> map = new HashMap<>();
        Node original = head;
        Node clone = new Node(head.val);
        map.put(head, clone);
        head = head.next;

        Node cloneHead = clone;
        Node answer = cloneHead;

        while (head != null) {
            clone.next = new Node(head.val);
            map.put(head, clone.next);
            clone = clone.next;
            head = head.next;
        }

        while (original != null) {
            if (original.random != null) {
                cloneHead.random = map.get(original.random);
            }
            original = original.next;
            cloneHead = cloneHead.next;
        }

        return answer;
    }


    public ListNode deleteDuplicates(ListNode head) {
        ListNode dummy = new ListNode(0, head);
        ListNode prev = dummy;

        while(head != null){
            if (head.next != null && head.val == head.next.val){
                while(head.next != null && head.val == head.next.val){
                    head = head.next;
                }
                prev.next = head.next;
            }
            else {
                prev = prev.next;
            }

            head = head.next;
        }

        return dummy.next;

    }

    public int kthSmallest(TreeNode root, int k) {

        List<Integer> nums = new ArrayList<Integer>();

        if(root == null){
            return 0;
        }
        
        traverser(root, nums);
        return nums.get(k -1 );
    }
        
    public void traverser(TreeNode root, List<Integer> nums){
        if (root == null){
            return;
        }
        traverser(root.left, nums);

        if (root != null){
            nums.add(Integer.valueOf(root.val));
        }

        traverser(root.right, nums);
    }
        
    class GoodNodes {
        public int ans = 0;
        public int goodNodes(TreeNode root) {
            traversal(root,root.val);
            return ans;
        }
    
        public void traversal(TreeNode root,int max){
    
            if(root == null){
                return ;
            }
    
            if(root.val >= max){
                ans++;
                max = root.val;  // for in between greater elements it is updated 
            }
    
            traversal(root.right,max);
            traversal(root.left,max);
        
        }
    }


    public ListNode reverseBetween(ListNode head, int left, int right) {
        ListNode solution = new ListNode(0, head);
        ListNode prev = solution;
        Deque<ListNode> deque = new LinkedList<>();

        if(head != null){

            for(int i = 0; i < left; i++){
                prev.next = head;
                head = head.next;
            }

            while(head != null){
                int j = left;
                if(j <= right){
                    deque.addFirst(head);
                }
                else {
                    deque.addLast(head);
                }
                head = head.next;
                j++;
            }

            while(!deque.isEmpty()){
                prev.next = deque.pop();
                prev = prev.next;
            }
        }

        return solution.next;
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> bigList = new ArrayList<>();
        if(root == null){
            return bigList;
        }

        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean isRightToLeft = false;

        while(!queue.isEmpty()){
            LinkedList<Integer> level = new LinkedList<>();
            int size = queue.size();

            for(int i = 0; i < size; i++){
                TreeNode current = queue.poll();
                // remember the following code is for each node in the level!

                if (isRightToLeft){
                    //add each element in the level from right to left
                    level.addFirst(current.val);
                }
                else {
                    //add each element in the level from left to right
                    level.addLast(current.val);
                }

                if(current.left != null){
                    queue.add(current.left);
                }

                if(current.right != null){
                    queue.add(current.right);
                }
            }
            bigList.add(level);
            isRightToLeft = !isRightToLeft;
        }

        return bigList;

    }


    public boolean isValidBST(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        int last = 0;
        boolean ans = true;

        if(root == null){
            return false;
        }

        checker(root, list);

        for(int i = 1; i < list.size(); i++){
            if(list.get(i) <= list.get(i - 1)){
                ans = false;
            }
        }

        return ans;
    }

    
    public void checker(TreeNode root, List<Integer> list){
        if(root == null){
            return;
        }

        
        checker(root.left, list);
        list.add(Integer.valueOf(root.val));
        checker(root.right, list);
        
        return;
    } 


    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode newHead = new ListNode(0, head);
        ListNode prev = newHead;
        ListNode curr = prev.next;
        ListNode saver2 = newHead.next;
        int count = 0;

        while(head != null){
            head = head.next;
            count++;
        }

        if(count == 1){
            return head;
        }

        if(count == 2 && n == 1){
            curr.next = null;
            return curr;
        }

        if(count == 2 && n == 2){
            return curr.next;
        }

        int i = 0;
        while(curr != null){

            prev = prev.next;
            curr = curr.next;

            if (i == count - n - 1 && prev.next.next == null){
                curr.next = null;
                break;
            }
            if (i == count - n - 1 && prev.next.next != null){
                prev.next = prev.next.next;
                break;
            }
            i++;

        }

        return saver2;

    }

    




}
