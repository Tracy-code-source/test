// 判断链表是否成环
#include <bits/stdc++.h>
using namespace std;
#define int long long
bool ret;
struct Info
{
    int data;
    struct Info* next;
};
struct Info* Create()
{
    struct Info* head = NULL;
    struct Info* current, *prev;
    current = prev = (struct Info*)malloc(sizeof(struct Info));
    if (head == NULL)
    {
        head = (struct Info*)malloc(sizeof(struct Info));
        head->next = current;
        prev = head;
    }
    cin >> current->data;
    while (current->data != 0)
    {
        prev->next = current;
        prev = current;
        current = (struct Info*)malloc(sizeof(struct Info));
        cin >> current->data;
    }
    prev->next = NULL;
    return head;
}
bool Check(struct Info* head)
{
    struct Info* quick = head;
    struct Info* slow = head;
    while (quick != NULL && quick->next != NULL)
    {
        quick = quick->next->next;
        slow = slow->next;
        if (quick == slow) return true;
    }
    return false;
}
signed main()
{
    struct Info* pre;
    pre = Create();
    pre = pre->next;
    ret = Check(pre);
    cout << ret << endl;
    system("pause");
    return 0;
}