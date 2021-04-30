class ListNode:
    def __init__(self, val: int):
        self.val = val
        self.next = None


class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        fast = head
        slow = self.reverseLinkedList(slow)
        while slow:
            if fast.val != slow.val:
                return False
            fast = fast.next
            slow = slow.next
        return True

    def reverseLinkedList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        while cur:
            post = cur.next
            cur.next = pre
            pre = cur
            cur = post
        return pre
