// 双向链表ADT
#include <bits/stdc++.h>
using namespace std;
#define int long long
#define check 999 //作为头节点数据校准值
struct Info
{
    struct Info* pri; // 指向前节点
    int data; //有效数据
    struct Info* next; //指向后节点
};
struct Info* Create()
{
    struct Info* head = NULL;
    struct Info *current, *prev;
    current = prev = (struct Info*)malloc(sizeof(struct Info)); // 头指针，流动指针，前节点指针
    if (head == NULL)
    {
        head = (struct Info*)malloc(sizeof(struct Info));
        head->data = check;
        head->next = current;
        head->pri = NULL;
        prev = head;
    }
    cin >> current->data;
    while (current->data != 0) // 以数据为零作为循环终止条件
    {
        prev->next = current;
        current->pri = prev; //前指针指向上个节点
        prev = current;
        current = (struct Info*)malloc(sizeof(struct Info));
        cin >> current->data;
    }
    prev->next = NULL;
    return head;
}
signed main()
{
    struct Info* pre, *ano;
    pre = ano = Create(); // ano记录头指针，方便下方倒序遍历
    pre = pre->next;
    // 从头开始遍历
    while (pre != NULL)
    {
        cout << pre->data << ' ';
        pre = pre->next;
    }
    cout << endl;
    // 从尾开始
    pre = ano;
    while (pre->next != NULL) pre = pre->next;
    while (pre != NULL)
    {
        if (pre->data != check) cout << pre->data << ' '; // 不打印头节点数据
        pre = pre->pri;
    }
    system("pause");
    return 0;
}
