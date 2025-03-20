// 反转链表
#include <bits/stdc++.h>
using namespace std;
#define int long long
struct Info
{
    int data;
    struct Info* next;
};
struct Info* Create() //创建链表
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
struct Info* Change(struct Info* head) //反转链表
{
    struct Info* prev = NULL; // 保证头节点next指向空
    struct Info* current = head->next; 
    struct Info* beh = current->next;
    while (beh != NULL)
    {
        current->next = prev;
        prev = current;
        current = beh; // current不断跟进
        beh = beh->next; // beh同时跟进
    }
    current->next = prev; // 最后一个节点指向前一个节点
    head->next = current;
    return head;
}
signed main()
{
    struct Info* pre;
    pre = Create();
    pre = Change(pre);
    pre = pre->next;
    while (pre != NULL)
    {
        cout << pre->data << ' ';
        pre = pre->next;
    }
    system("pause");
    return 0;
}
