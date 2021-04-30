class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next = None


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummpy = slow = fast = ListNode(-1)
        fast.next = head
        for _ in range(n + 1):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummpy.next
