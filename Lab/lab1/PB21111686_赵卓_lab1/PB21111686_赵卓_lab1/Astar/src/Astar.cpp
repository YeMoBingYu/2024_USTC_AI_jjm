#include <vector>
#include <iostream>
#include <queue>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
using namespace std;

struct Map_Cell
{
    int type;
};

struct Search_Cell
{
    int h;
    int g;
    int x;      // 水平位置
    int y;      // 竖直位置
    string way; // 路径
    int energy; // 体力
};

// 自定义比较函数对象，按照 Search_Cell 结构体的 g + h 属性进行比较
struct CompareF
{
    bool operator()(const Search_Cell *a, const Search_Cell *b) const
    {
        return (a->g + a->h) > (b->g + b->h); // 较小的 g + h 值优先级更高
    }
};

// 启发式函数：曼哈顿距离
int Heuristic_Funtion(pair<int, int> &end_point, Search_Cell *current)
{
    return abs(current->x - end_point.first) + abs(current->y - end_point.second);
}

void Astar_search(const string input_file, int &step_nums, string &way)
{
    // 加载地图
    ifstream file(input_file);
    if (!file.is_open())
    {
        cout << "Error opening file!" << endl;
        return;
    }

    string line;
    getline(file, line); // 读取第一行
    stringstream ss(line);
    string word;
    vector<string> words;
    while (ss >> word)
    {
        words.push_back(word);
    }
    int M = stoi(words[0]);
    int N = stoi(words[1]);
    int T = stoi(words[2]);

    pair<int, int> start_point; // 起点
    pair<int, int> end_point;   // 终点
    Map_Cell **Map = new Map_Cell *[M];
    for (int i = 0; i < M; i++)
    {
        Map[i] = new Map_Cell[N];
        getline(file, line);
        stringstream ss(line);
        string word;
        vector<string> words;
        while (ss >> word)
        {
            words.push_back(word);
        }
        for (int j = 0; j < N; j++)
        {
            Map[i][j].type = stoi(words[j]);
            if (Map[i][j].type == 3)
            {
                start_point = {i, j};
            }
            else if (Map[i][j].type == 4)
            {
                end_point = {i, j};
            }
        }
    }

    // A*搜索
    Search_Cell *search_cell = new Search_Cell;
    search_cell->x = start_point.first;
    search_cell->y = start_point.second;
    search_cell->g = 0;
    search_cell->h = 0;
    search_cell->energy = T;
    search_cell->way = "";

    priority_queue<Search_Cell *, vector<Search_Cell *>, CompareF> open_list;
    vector<Search_Cell *> close_list;
    open_list.push(search_cell);

    vector<vector<int>> visit(M + 1, vector<int>(N + 1, 0));
    visit[start_point.first][start_point.second] = 1;

    while (!open_list.empty())
    {
        Search_Cell *current = open_list.top();
        open_list.pop();
        visit[current->x][current->y] = 1;
        if (current->x == end_point.first && current->y == end_point.second)
        {
            step_nums = current->g;
            way = current->way;
            break;
        }
        if (current->energy <= 0)
            continue;
        vector<pair<int, int>> location = {{current->x + 1, current->y}, {current->x - 1, current->y}, 
        {current->x, current->y - 1}, {current->x, current->y + 1}};
        for (auto &loc : location)
        {
            int i = loc.first, j = loc.second;
            if (i >= 0 && i < M && j >= 0 && j < N && Map[i][j].type != 1 && visit[i][j] == 0)
            {
                Search_Cell *next = new Search_Cell;
                next->x = i;
                next->y = j;
                next->g = current->g + 1;
                next->h = Heuristic_Funtion(end_point, next);
                if (Map[i][j].type == 2)
                    next->energy = T;
                else
                    next->energy = current->energy - 1;
                if (i == current->x + 1)
                    next->way = current->way + "D";
                else if (i == current->x - 1)
                    next->way = current->way + "U";
                else if (j == current->y + 1)
                    next->way = current->way + "R";
                else if (j == current->y - 1)
                    next->way = current->way + "L";
                open_list.push(next);
            }
        }
        close_list.push_back(current);
    }

    // 释放动态内存
    for (int i = 0; i < M; i++)
    {
        delete Map[i];
    }
    delete Map;
    while (!open_list.empty())
    {
        auto temp = open_list.top();
        delete temp;
        open_list.pop();
    }
    for (int i = 0; i < close_list.size(); i++)
    {
        delete close_list[i];
    }

    return;
}

void output(const string output_file, int &step_nums, string &way)
{
    ofstream file(output_file);
    if (file.is_open())
    {
        file << step_nums << endl;
        if (step_nums >= 0)
        {
            file << way << endl;
        }

        file.close();
    }
    else
    {
        cerr << "Can not open file: " << output_file << endl;
    }
    return;
}

int main(int argc, char *argv[])
{
    string input_base = "../input/input_";
    string output_base = "../output/output_";
    // input_0为讲义样例，此处不做测试
    for (int i = 1; i < 11; i++)
    {
        int step_nums = -1;
        string way = "";
        Astar_search(input_base + to_string(i) + ".txt", step_nums, way);
        output(output_base + to_string(i) + ".txt", step_nums, way);
    }
    return 0;
}