#include <vector>
#include <map>
#include <limits>
#include <iostream>
#include <string>

namespace ChineseChess
{
    // 棋力评估，这里的棋盘方向和输入棋盘方向不同，在使用时需要仔细
    // 生成合法动作代码部分已经使用，经过测试是正确的，大家可以参考
    std::vector<std::vector<int>> JiangPosition = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {5, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {1, -8, -9, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> ShiPosition = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> XiangPosition = {
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 3, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, -2, 0, 0, 0, 0, 0, 0, 0},
    };

    std::vector<std::vector<int>> MaPosition = {
        {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
        {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
        {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
        {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
        {2, -10, 4, 10, 15, 16, 12, 11, 6, 2},
        {0, 5, 7, 7, 14, 15, 19, 15, 9, 8},
        {2, 4, 6, 10, 13, 11, 12, 11, 15, 2},
        {-3, 2, 4, 6, 10, 12, 20, 10, 8, 2},
        {0, -3, 5, 4, 2, 2, 5, 4, 2, 2},
    };

    std::vector<std::vector<int>> PaoPosition = {
        {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
        {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
        {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
        {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
        {3, 2, 5, 0, 4, 4, 4, -4, -7, -6},
        {3, 2, 3, 0, 0, 0, 2, -5, -4, -5},
        {1, 2, 4, 0, 3, 0, 3, 0, 0, 0},
        {0, 1, 0, 0, 0, 0, 3, 1, 2, 4},
        {0, 0, 1, 0, -1, 0, 0, 1, 2, 4},
    };

    std::vector<std::vector<int>> JuPosition = {
        {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
        {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
        {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {0, 0, 12, 14, 15, 15, 16, 16, 33, 14},
        {12, 12, 12, 12, 14, 14, 16, 14, 16, 13},
        {4, 6, 4, 4, 12, 11, 13, 7, 9, 7},
        {6, 8, 8, 9, 12, 11, 13, 8, 12, 8},
        {-6, 5, -2, 4, 8, 8, 6, 6, 6, 6},
    };

    std::vector<std::vector<int>> BingPosition = {
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
        {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
        {0, 0, 0, 6, 7, 40, 42, 55, 70, 4},
        {0, 0, 0, 0, 0, 35, 40, 55, 65, 2},
        {0, 0, 0, -2, 4, 22, 30, 45, 50, 0},
        {0, 0, 0, 0, 0, 18, 27, 30, 30, 0},
        {0, 0, 0, -2, 3, 10, 20, 20, 20, 0},
    };

    // 棋子价值评估
    std::map<std::string, int> piece_values = {
        {"Jiang", 10000},
        {"Shi", 10},
        {"Xiang", 30},
        {"Ma", 300},
        {"Ju", 500},
        {"Pao", 300},
        {"Bing", 90}};

    // 行期可能性评估，对下一步动作的评估
    std::map<std::string, int> next_move_values = {
        {"Jiang", 9999},
        {"Ma", 100},
        {"Ju", 500},
        {"Pao", 100},
        {"Xiang", 30},
        {"Shi", 10},
        {"Bing", -20}};

    // 动作结构体，每个动作设置score，可以方便剪枝
    struct Move
    {
        int init_x;
        int init_y;
        int next_x;
        int next_y;
        int score;
    };

    // 定义棋盘上的棋子结构体
    struct ChessPiece
    {
        char name;          // 棋子名称
        int init_x, init_y; // 棋子的坐标
        bool color;         // 棋子阵营 true为红色、false为黑色
    };

    // 定义棋盘类
    class ChessBoard
    {
    private:
        int sizeX, sizeY;                     // 棋盘大小，固定
        std::vector<ChessPiece> pieces;       // 棋盘上所有棋子
        std::vector<std::vector<char>> board; // 当前棋盘、二维数组表示
        std::vector<Move> red_moves;          // 红方棋子的合法动作
        std::vector<Move> black_moves;        // 黑方棋子的合法动作
    public:
        // 初始化棋盘，提取棋盘上棋子，并生成所有合法动作
        void initializeBoard(const std::vector<std::vector<char>> &init_board)
        {
            board = init_board;
            sizeX = board.size();
            sizeY = board[0].size();

            for (int i = 0; i < sizeX; ++i)
            {
                for (int j = 0; j < sizeY; ++j)
                {
                    char pieceChar = board[i][j];
                    if (pieceChar == '.')
                        continue;

                    ChessPiece piece;
                    piece.init_x = j;
                    piece.init_y = i;
                    piece.color = (pieceChar >= 'A' && pieceChar <= 'Z');
                    piece.name = pieceChar;
                    pieces.push_back(piece);

                    switch (pieceChar)
                    {
                    case 'R':
                        generateJuMoves(j, i, piece.color);
                        break;
                    case 'C':
                        generatePaoMoves(j, i, piece.color);
                        break;
                    case 'N':
                        generateMaMoves(j, i, piece.color);
                        break;
                    case 'B':
                        generateXiangMoves(j, i, piece.color);
                        break;
                    case 'A':
                        generateShiMoves(j, i, piece.color);
                        break;
                    case 'K':
                        generateJiangMoves(j, i, piece.color);
                        break;
                    case 'P':
                        generateBingMoves(j, i, piece.color);
                        break;
                    case 'r':
                        generateJuMoves(j, i, piece.color);
                        break;
                    case 'c':
                        generatePaoMoves(j, i, piece.color);
                        break;
                    case 'n':
                        generateMaMoves(j, i, piece.color);
                        break;
                    case 'b':
                        generateXiangMoves(j, i, piece.color);
                        break;
                    case 'a':
                        generateShiMoves(j, i, piece.color);
                        break;
                    case 'k':
                        generateJiangMoves(j, i, piece.color);
                        break;
                    case 'p':
                        generateBingMoves(j, i, piece.color);
                        break;
                    default:
                        break;
                    }
                }
            }
        }

        // 生成车的合法动作
        void generateJuMoves(int x, int y, bool color)
        {
            // 前后左右分别进行搜索，遇到棋子停止，不同阵营可以吃掉
            std::vector<Move> JuMoves;
            for (int i = x + 1; i < sizeY; i++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.')
                {
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                else
                    JuMoves.push_back(cur_move);
            }

            for (int i = x - 1; i >= 0; i--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                cur_move.score = 0;
                if (board[y][i] != '.')
                {
                    bool cur_color = (board[y][i] >= 'A' && board[y][i] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                else
                    JuMoves.push_back(cur_move);
            }

            for (int j = y + 1; j < sizeX; j++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.')
                {
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                else
                    JuMoves.push_back(cur_move);
            }

            for (int j = y - 1; j >= 0; j--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = j;
                cur_move.score = 0;
                if (board[j][x] != '.')
                {
                    bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                    if (cur_color != color)
                    {
                        JuMoves.push_back(cur_move);
                    }
                    break;
                }
                else
                    JuMoves.push_back(cur_move);
            }
            for (int i = 0; i < JuMoves.size(); i++)
            {
                if (color)
                {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_x][9 - JuMoves[i].next_y] - JuPosition[x][9 - y];
                    red_moves.push_back(JuMoves[i]);
                }
                else
                {
                    JuMoves[i].score = JuPosition[JuMoves[i].next_x][JuMoves[i].next_y] - JuPosition[x][y];
                    black_moves.push_back(JuMoves[i]);
                }
            }
        }

        // 生成马的合法动作
        void generateMaMoves(int x, int y, bool color)
        {
            std::vector<Move> MaMoves;
            int dx[] = {2, 1, -1, -2, -2, -1, 1, 2};
            int dy[] = {1, 2, 2, 1, -1, -2, -2, -1};
            int wrong_x[] = {1, 0, 0, -1, -1, 0, 0, 1};
            int wrong_y[] = {0, 1, 1, 0, 0, -1, -1, 0};
            for (int i = 0; i < 8; i++)
            {
                Move cur_move;
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx < 0 || nx >= 9 || ny < 0 || ny >= 10)
                    continue;
                int wx = x + wrong_x[i];
                int wy = y + wrong_y[i];
                if (board[wy][wx] != '.')
                    continue;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                cur_move.score = 0;
                if (board[ny][nx] != '.')
                {
                    bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                    if (cur_color != color)
                        MaMoves.push_back(cur_move);
                }
                else
                    MaMoves.push_back(cur_move);
            }
            for (int i = 0; i < MaMoves.size(); i++)
            {
                if (color)
                {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_x][9 - MaMoves[i].next_y] - MaPosition[x][9 - y];
                    red_moves.push_back(MaMoves[i]);
                }
                else
                {
                    MaMoves[i].score = MaPosition[MaMoves[i].next_x][MaMoves[i].next_y] - MaPosition[x][y];
                    black_moves.push_back(MaMoves[i]);
                }
            }
        }

        // 生成炮的合法动作
        void generatePaoMoves(int x, int y, bool color)
        {
            // 和车生成动作相似，需要考虑炮翻山吃子的情况
            std::vector<Move> PaoMoves;
            for (int i = x + 1; i < 9; i++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                if (board[y][i] != '.')
                {
                    for (int j = i + 1; j < 9; j++)
                    {
                        bool cur_color = (board[y][j] >= 'A' && board[y][j] <= 'Z');
                        if (cur_color != color)
                        {
                            cur_move.next_x = j;
                            PaoMoves.push_back(cur_move);
                            break;
                        }
                    }
                    break;
                }
                else
                    PaoMoves.push_back(cur_move);
            }

            for (int i = x - 1; i >= 0; i--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = i;
                cur_move.next_y = y;
                if (board[y][i] != '.')
                {
                    for (int j = i - 1; j >= 0; j--)
                    {
                        bool cur_color = (board[y][j] >= 'A' && board[y][j] <= 'Z');
                        if (cur_color != color)
                        {
                            cur_move.next_x = j;
                            PaoMoves.push_back(cur_move);
                            break;
                        }
                    }
                    break;
                }
                else
                    PaoMoves.push_back(cur_move);
            }

            for (int i = y + 1; i < 10; i++)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = i;
                if (board[i][x] != '.')
                {
                    for (int j = i + 1; j < 10; j++)
                    {
                        bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                        if (cur_color != color)
                        {
                            cur_move.next_y = j;
                            PaoMoves.push_back(cur_move);
                            break;
                        }
                    }
                    break;
                }
                else
                    PaoMoves.push_back(cur_move);
            }

            for (int i = y - 1; i >= 0; i--)
            {
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = x;
                cur_move.next_y = i;
                if (board[i][x] != '.')
                {
                    for (int j = i - 1; j >= 0; j--)
                    {
                        bool cur_color = (board[j][x] >= 'A' && board[j][x] <= 'Z');
                        if (cur_color != color)
                        {
                            cur_move.next_y = j;
                            PaoMoves.push_back(cur_move);
                            break;
                        }
                    }
                    break;
                }
                else
                    PaoMoves.push_back(cur_move);
            }

            for (int i = 0; i < PaoMoves.size(); i++)
            {
                if (color)
                {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][9 - PaoMoves[i].next_y] - PaoPosition[x][9 - y];
                    red_moves.push_back(PaoMoves[i]);
                }
                else
                {
                    PaoMoves[i].score = PaoPosition[PaoMoves[i].next_x][PaoMoves[i].next_y] - PaoPosition[x][y];
                    black_moves.push_back(PaoMoves[i]);
                }
            }
        }

        // 生成相的合法动作
        void generateXiangMoves(int x, int y, bool color)
        {
            std::vector<Move> XiangMoves;
            int dx[] = {2, 2, -2, -2};
            int dy[] = {2, -2, 2, -2};
            int wrong_x[] = {1, 1, -1, -1};
            int wrong_y[] = {1, -1, 1, -1};

            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                int wx = x + wrong_x[i];
                int wy = y + wrong_y[i];
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                if (color)
                {
                    if (nx >= 0 && nx < 9 && ny >= 5 && ny <= 9 && board[wy][wx] == '.')
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                XiangMoves.push_back(cur_move);
                        }
                        else
                            XiangMoves.push_back(cur_move);
                    }
                }
                else
                {
                    if (nx >= 0 && nx < 9 && ny >= 0 && ny <= 4 && board[wy][wx] == '.')
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                XiangMoves.push_back(cur_move);
                        }
                        else
                            XiangMoves.push_back(cur_move);
                    }
                }
            }
            for (int i = 0; i < XiangMoves.size(); i++)
            {
                if (color)
                {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][9 - XiangMoves[i].next_y] - XiangPosition[x][9 - y];
                    red_moves.push_back(XiangMoves[i]);
                }
                else
                {
                    XiangMoves[i].score = XiangPosition[XiangMoves[i].next_x][XiangMoves[i].next_y] - XiangPosition[x][y];
                    black_moves.push_back(XiangMoves[i]);
                }
            }
        }

        // 生成士的合法动作
        void generateShiMoves(int x, int y, bool color)
        {
            std::vector<Move> ShiMoves;
            int dx[] = {1, 1, -1, -1};
            int dy[] = {1, -1, 1, -1};
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                if (color)
                {
                    if (nx >= 3 && nx <= 5 && ny >= 7 && ny <= 9)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                ShiMoves.push_back(cur_move);
                        }
                        else
                            ShiMoves.push_back(cur_move);
                    }
                }
                else
                {
                    if (nx >= 3 && nx <= 5 && ny >= 0 && ny <= 2)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                ShiMoves.push_back(cur_move);
                        }
                        else
                            ShiMoves.push_back(cur_move);
                    }
                }
            }
            for (int i = 0; i < ShiMoves.size(); i++)
            {
                if (color)
                {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][9 - ShiMoves[i].next_y] - ShiPosition[x][9 - y];
                    red_moves.push_back(ShiMoves[i]);
                }
                else
                {
                    ShiMoves[i].score = ShiPosition[ShiMoves[i].next_x][ShiMoves[i].next_y] - ShiPosition[x][y];
                    black_moves.push_back(ShiMoves[i]);
                }
            }
        }

        // 生成将的合法动作
        void generateJiangMoves(int x, int y, bool color)
        {
            std::vector<Move> JiangMoves;
            int dx[] = {1, 0, 0, -1};
            int dy[] = {0, 1, -1, 0};
            for (int i = 0; i < 4; i++)
            {
                int nx = x + dx[i];
                int ny = y + dy[i];
                Move cur_move;
                cur_move.init_x = x;
                cur_move.init_y = y;
                cur_move.next_x = nx;
                cur_move.next_y = ny;
                if (color)
                {
                    if (nx >= 3 && nx <= 5 && ny >= 7 && ny <= 9)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                JiangMoves.push_back(cur_move);
                        }
                        else
                            JiangMoves.push_back(cur_move);
                    }
                }
                else
                {
                    if (nx >= 3 && nx <= 5 && ny >= 0 && ny <= 2)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                JiangMoves.push_back(cur_move);
                        }
                        else
                            JiangMoves.push_back(cur_move);
                    }
                }
            }
            for (int i = 0; i < JiangMoves.size(); i++)
            {
                if (color)
                {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][9 - JiangMoves[i].next_y] - JiangPosition[x][9 - y];
                    red_moves.push_back(JiangMoves[i]);
                }
                else
                {
                    JiangMoves[i].score = JiangPosition[JiangMoves[i].next_x][JiangMoves[i].next_y] - JiangPosition[x][y];
                    black_moves.push_back(JiangMoves[i]);
                }
            }
        }

        // 生成兵的合法动作
        void generateBingMoves(int x, int y, bool color)
        {
            // 需要分条件考虑，小兵在过楚河汉界之前只能前进，之后可以左右前
            std::vector<Move> BingMoves;
            int dx_red[] = {0, -1, 1};
            int dy_red[] = {-1, 0, 0};
            int dx_black[] = {0, -1, 1};
            int dy_black[] = {1, 0, 0};
            if (color)
            {
                if (y >= 5)
                {
                    Move cur_move;
                    int nx = x;
                    int ny = y - 1;
                    cur_move.init_x = x;
                    cur_move.init_y = y;
                    cur_move.next_x = x;
                    cur_move.next_y = y - 1;
                    if (nx >= 0 && nx < 9 && ny >= 0 && ny < 10)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                BingMoves.push_back(cur_move);
                        }
                        else
                            BingMoves.push_back(cur_move);
                    }
                }
                else
                {
                    for (int i = 0; i < 3; i++)
                    {
                        int nx = x + dx_red[i];
                        int ny = y + dy_red[i];
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        if (nx >= 0 && nx < 9 && ny >= 0 && ny < 10)
                        {
                            if (board[ny][nx] != '.')
                            {
                                bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                                if (cur_color != color)
                                    BingMoves.push_back(cur_move);
                            }
                            else
                                BingMoves.push_back(cur_move);
                        }
                    }
                }
            }
            else
            {
                if (y <= 4)
                {
                    Move cur_move;
                    int nx = x;
                    int ny = y + 1;
                    cur_move.init_x = x;
                    cur_move.init_y = y;
                    cur_move.next_x = x;
                    cur_move.next_y = y + 1;
                    if (nx >= 0 && nx < 9 && ny >= 0 && ny < 10)
                    {
                        if (board[ny][nx] != '.')
                        {
                            bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                            if (cur_color != color)
                                BingMoves.push_back(cur_move);
                        }
                        else
                            BingMoves.push_back(cur_move);
                    }
                }
                else
                {
                    for (int i = 0; i < 3; i++)
                    {
                        int nx = x + dx_black[i];
                        int ny = y + dy_black[i];
                        Move cur_move;
                        cur_move.init_x = x;
                        cur_move.init_y = y;
                        cur_move.next_x = nx;
                        cur_move.next_y = ny;
                        if (nx >= 0 && nx < 9 && ny >= 0 && ny < 10)
                        {
                            if (board[ny][nx] != '.')
                            {
                                bool cur_color = (board[ny][nx] >= 'A' && board[ny][nx] <= 'Z');
                                if (cur_color != color)
                                    BingMoves.push_back(cur_move);
                            }
                            else
                                BingMoves.push_back(cur_move);
                        }
                    }
                }
            }
            for (int i = 0; i < BingMoves.size(); i++)
            {
                if (color)
                {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_x][9 - BingMoves[i].next_y] - BingPosition[x][9 - y];
                    red_moves.push_back(BingMoves[i]);
                }
                else
                {
                    BingMoves[i].score = BingPosition[BingMoves[i].next_x][BingMoves[i].next_y] - BingPosition[x][y];
                    black_moves.push_back(BingMoves[i]);
                }
            }
        }

        // 判断当前棋局是否结束
        bool judgeTermination()
        {
            int count = 0;
            for (auto &piece : pieces)
            {
                if (piece.name == 'k' || piece.name == 'K')
                    count++;
            }
            if (count == 1)
                return true;
            return false;
        }

        // 棋盘分数评估，根据当前棋盘进行棋子价值和棋力评估，max玩家减去min玩家分数
        int evaluateNode()
        {
            int MAX = 0, MIN = 0;
            // 棋子价值评估
            for (auto &piece : pieces)
            {
                switch (piece.name)
                {
                case 'R':
                    MAX += piece_values["Ju"];
                    break;
                case 'C':
                    MAX += piece_values["Pao"];
                    break;
                case 'N':
                    MAX += piece_values["Ma"];
                    break;
                case 'B':
                    MAX += piece_values["Xiang"];
                    break;
                case 'A':
                    MAX += piece_values["Shi"];
                    break;
                case 'K':
                    MAX += piece_values["Jiang"];
                    break;
                case 'P':
                    MAX += piece_values["Bing"];
                    break;
                case 'r':
                    MIN += piece_values["Ju"];
                    break;
                case 'c':
                    MIN += piece_values["Pao"];
                    break;
                case 'n':
                    MIN += piece_values["Ma"];
                    break;
                case 'b':
                    MIN += piece_values["Xiang"];
                    break;
                case 'a':
                    MIN += piece_values["Shi"];
                    break;
                case 'k':
                    MIN += piece_values["Jiang"];
                    break;
                case 'p':
                    MIN += piece_values["Bing"];
                    break;
                default:
                    break;
                }
            }
            // 棋力评估
            for (int i = 0; i < sizeX; i++)
            {
                for (int j = 0; j < sizeY; j++)
                {
                    switch (board[i][j])
                    {
                    case 'R':
                        MAX += JuPosition[j][9 - i];
                        break;
                    case 'C':
                        MAX += PaoPosition[j][9 - i];
                        break;
                    case 'N':
                        MAX += MaPosition[j][9 - i];
                        break;
                    case 'B':
                        MAX += XiangPosition[j][9 - i];
                        break;
                    case 'A':
                        MAX += ShiPosition[j][9 - i];
                        break;
                    case 'K':
                        MAX += JiangPosition[j][9 - i];
                        break;
                    case 'P':
                        MAX += BingPosition[j][9 - i];
                        break;
                    case 'r':
                        MIN += JuPosition[j][i];
                        break;
                    case 'c':
                        MIN += PaoPosition[j][i];
                        break;
                    case 'n':
                        MIN += MaPosition[j][i];
                        break;
                    case 'b':
                        MIN += XiangPosition[j][i];
                        break;
                    case 'a':
                        MIN += ShiPosition[j][i];
                        break;
                    case 'k':
                        MIN += JiangPosition[j][i];
                        break;
                    case 'p':
                        MIN += BingPosition[j][i];
                        break;
                    default:
                        break;
                    }
                }
            }
            return MAX - MIN;
        }

        // 得到当前的棋子，动作，和棋盘状态
        std::vector<Move> getMoves(bool color)
        {
            if (color)
                return red_moves;
            return black_moves;
        }

        std::vector<ChessPiece> getChessPiece()
        {
            return pieces;
        }

        std::vector<std::vector<char>> getBoard()
        {
            return board;
        }
    };

    // 博弈树节点
    class GameTreeNode
    {
    private:
        bool color;              // 当前玩家类型
        ChessBoard board;        // 当前棋盘状态
        Move move;               // 得到的move
        int evaluationScore = 0; // 棋盘评估分数

    public:
        std::vector<GameTreeNode *> children; // 子节点
        GameTreeNode(bool color, std::vector<std::vector<char>> initBoard, Move move)
            : color(color), move(move)
        {
            board.initializeBoard(initBoard);
            children.clear();
            evaluationScore = board.evaluateNode();
        }

        // 根据当前棋盘和动作构建子节点
        GameTreeNode *updateBoard(std::vector<std::vector<char>> cur_board, Move move, bool color)
        {
            cur_board[move.next_y][move.next_x] = cur_board[move.init_y][move.init_x];
            cur_board[move.init_y][move.init_x] = '.';
            GameTreeNode *child_node = new GameTreeNode(color, cur_board, move);
            return child_node;
        }

        // 返回评估分数
        int getEvaluationScore()
        {
            return evaluationScore;
        }
        void setEvaluationScore(int score)
        {
            evaluationScore = score;
        }
        bool getcolor()
        {
            return color;
        }
        Move get_move()
        {
            return move;
        }
        // 返回棋盘
        ChessBoard getBoardClass()
        {
            return board;
        }
        std::vector<GameTreeNode *> &get_children()
        {
            return children;
        }
        // 注销
        ~GameTreeNode()
        {
            for (GameTreeNode *child : children)
            {
                delete child;
            }
        }
    };
}