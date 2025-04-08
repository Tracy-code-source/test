//排序
#include <bits/stdc++.h>
using namespace std;
#define int long long
// 插入排序
void insert_sort(int a[], int n) // n方
{
    for (int i = 1; i < n; i++) 
    {
        int temp = a[i], j = i - 1;
        while (j >= 0 && a[j] > temp) 
        {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = temp;
    }
}
// 归并排序
void merge(int a[], int l, int mid, int r) 
{
    int i = l, j = mid + 1, k = 0;  // i, j, k分别指向左半边、右半边和临时数组的索引
    int b[r - l + 1];               // 创建临时数组
    while (i <= mid && j <= r)      // 当两个子数组都有元素时
    {
        if (a[i] < a[j])            // 如果左半边的元素小于右半边的元素
        {
            b[k++] = a[i++];        // 将左半边的元素放入临时数组
        } 
        else if (a[i] > a[j])       // 如果左半边的元素大于右半边的元素
        {
            b[k++] = a[j++];        // 将右半边的元素放入临时数组
        } 
    }
    while (i <= mid)                // 当左半边还有元素时
    {
        b[k++] = a[i++];            
    }
    while (j <= r)                  // 当右半边还有元素时
    {
        b[k++] = a[j++];
    }
    for (int i = l; i <= r; i++)    // 将临时数组中的元素复制回原数组
    {
        a[i] = b[i - l];
    }
}
void merge_sort(int a[], int l, int r) 
{
    if (l >= r) return;             // 递归终止条件
    int mid = (l + r) / 2;
    merge_sort(a, l, mid);          // 递归排序左半边
    merge_sort(a, mid + 1, r);      // 递归排序右半边
    merge(a, l, mid, r);            // 合并两个已排序的子数组
}
// 快速排序
int partition(int a[], int l, int r) //板子
{
    int pivot = a[l], i = l + 1, j = r;
    while (i <= j) 
    {
        while (i <= r && a[i] < pivot) i++;
        while (j >= l && a[j] > pivot) j--;
        if (i < j) swap(a[i++], a[j--]);
    }
    swap(a[l], a[j]);
    return j;
}
void quick_sort(int a[], int l, int r) 
{
    if (l >= r) return;
    int pivot = partition(a, l, r);
    quick_sort(a, l, pivot - 1);
    quick_sort(a, pivot + 1, r);
}
// 计数排序
void count_sort(int a[], int n) 
{
    int maxnum = -9999999;
    for (int i = 0; i < n; i++)  
    {
        maxnum = max(maxnum, a[i]);  // 找到最大值
    }
    int count[maxnum + 1] = {0};   // 创建计数数组
    for (int i = 0; i < n; i++) 
    {
        count[a[i]]++;              // 统计每个元素的出现次数
    }
    int index = 0;
    for (int i = 0; i <= maxnum; i++) 
    {
        while (count[i] > 0) 
        {
            a[index++] = i;         // 将元素放回原数组
            count[i]--;
        }
    }
}
// 基数计数排序
void radix_count_sort(int a[], int n, int exp) 
{
    vector<int> output(n);
    int count[10] = {0};
    for (int i = 0; i < n; i++) 
    {
        count[(a[i] / exp) % 10]++;
    }
    for (int i = 1; i < 10; i++) 
    {
        count[i] += count[i - 1];
    }
    for (int i = n - 1; i >= 0; i--) 
    {
        output[count[(a[i] / exp) % 10] - 1] = a[i];
        count[(a[i] / exp) % 10]--;
    }
    for (int i = 0; i < n; i++) 
    {
        a[i] = output[i];
    }
}
// 功能选择
void sort(int a[], int n, int method) 
{
    switch (method) {
        case 1:
            insert_sort(a, n);
            break;
        case 2:
            merge_sort(a, 0, n - 1);
            break;
        case 3:
            quick_sort(a, 0, n - 1);
            break;
        case 4:
            count_sort(a, n);
            break;
        case 5:
            for (int exp = 1; exp <= *max_element(a, a + n); exp *= 10) 
            {
                radix_count_sort(a, n, exp);
            }
            break;
        default:
            cout << "ERROR" << endl;
    }
}
signed main() {
    int n, method;
    cout << "Please enter the num of elements: ";
    cin >> n;
    int a[n];
    cout << "Enter your elements: ";
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }
    cout << "Please choose your mode (1): Insert  (2): Merge  (3): Quick  (4): Count  (5): Radix: ";
    cin >> method;
    sort(a, n, method);
    cout << "Sort Completed: ";
    for (int i = 0; i < n; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
    system("pause");
    return 0;
}