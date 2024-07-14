#include <fstream>
#include <string>
#include "node.h"

using namespace ChineseChess;

int max(int a, int b)
{
    return a > b ? a : b;
}

int min(int a, int b)
{
    return a < b ? a : b;
}
// α-β减枝搜索
int alphaBeta(GameTreeNode *node, int alpha, int beta, int depth)
{
    if (depth == 0)
    {
        return node->getEvaluationScore();
    }
    else
    {
        ChessBoard board = node->getBoardClass();
        std::vector<Move> moves = board.getMoves(node->getcolor());
        std::vector<std::vector<char>> cur_board = board.getBoard();
        if (node->getcolor())
        {
            for (auto &move : moves)
            {
                GameTreeNode *child_node = node->updateBoard(cur_board, move, !node->getcolor());
                node->children.push_back(child_node);
                alpha = max(alpha, alphaBeta(child_node, alpha, beta, depth - 1));
                if (alpha >= beta)
                    break;
            }
            node->setEvaluationScore(alpha);
            return alpha;
        }
        else
        {
            for (auto &move : moves)
            {
                GameTreeNode *child_node = node->updateBoard(cur_board, move, !node->getcolor());
                node->children.push_back(child_node);
                beta = min(beta, alphaBeta(child_node, alpha, beta, depth - 1));
                if (alpha >= beta)
                    break;
            }
            node->setEvaluationScore(beta);
            return beta;
        }
    }
    return 0;
}

void output(std::string name, char piece, std::string &location, std::string &location_)
{
    std::ofstream file(name);
    if (file.is_open())
    {
        file << piece << location.c_str() << location_.c_str() << std::endl;
        file.close();
    }
    else
    {
        std::cerr << "Can not open file" << std::endl;
    }
}

int main()
{
    for (int k = 1; k <= 10; k++)
    {
        // 加载当前棋局
        std::ifstream file("../input/" + std::to_string(k) + ".txt");
        if (!file.is_open())
        {
            std::cout << "Error opening file!" << std::endl;
            return -1;
        }
        std::vector<std::vector<char>> board;
        std::string line;
        int n = 0;
        while (std::getline(file, line))
        {
            std::vector<char> row;
            for (char ch : line)
            {
                row.push_back(ch);
            }
            board.push_back(row);
            n++;
            if (n >= 10)
                break;
        }
        file.close();
        // 四层博弈树，预测四步
        Move move;
        GameTreeNode *root = new GameTreeNode(true, board, move);
        int max = alphaBeta(root, -10000, 10000, 4);
        for (auto &child : root->get_children())
        {
            if (child->getEvaluationScore() == max)
            {
                move = child->get_move();
                break;
            }
        }
        // 输出
        std::string location = "(" + std::to_string(move.init_x) + "," + std::to_string(move.init_y) + ")";
        std::string location_ = "(" + std::to_string(move.next_x) + "," + std::to_string(move.next_y) + ")";
        std::string file_name = "../output/output_" + std::to_string(k) + ".txt";
        output(file_name, board[move.init_y][move.init_x], location, location_);
        // std::cout << board[move.init_y][move.init_x] << location << location_ << std::endl;
        delete root;
    }
    return 0;
}