//寻找链表中点
#include <bits/stdc++.h>
using namespace std;
#define int long long
struct Info
{
    int data;
    struct Info* next;
};
struct Info* Create()
{
    struct Info* head = NULL; // 头指针
    struct Info* current, *prev; // 流动指针, 前节点指针
    current = prev = (struct Info*)malloc(sizeof(struct Info));
    if (head == NULL) //头指针不包含有效值， 方便后续统一操作
    {
        head = (struct Info*)malloc(sizeof(struct Info));
        head->next = current;
        prev = head;
    }
    cin >> current->data;
    while (current->data != 0) //以数据为零作为循环终点
    {
        prev->next = current;
        prev = current;
        current = (struct Info*)malloc(sizeof(struct Info)); // 重新为流动指针分配空间
        cin >> current->data;
    }
    prev->next = NULL; // 尾指针指向空
    return head; //返回头指针
}
struct Info* Mid(struct Info* head)
{
    struct Info* quick = head;
    struct Info* slow = head;
    while (quick != NULL && quick->next != NULL)
    {
        quick = quick->next->next;
        slow = slow->next;
    }
    return slow;
}
signed main()
{
    int ret;
    struct Info* pre; // 声明一个结构体指针
    pre = Create(); //创建链表
    pre = pre->next;
    ret = Mid(pre)->data;
    cout << ret << endl;
    system("pause");
    return 0;
}