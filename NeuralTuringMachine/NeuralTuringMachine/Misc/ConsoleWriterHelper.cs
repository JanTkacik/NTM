using System;

namespace NeuralTuringMachine.Misc
{
    public static class ConsoleWriterHelper
    {
        public static Tuple<int, int> LogDoubleTable(string tableName, double[][] array, int posX, int posY)
        {
            Console.SetCursorPosition(posX, posY);
            Console.ForegroundColor = ConsoleColor.White;

            int xLength = array.Length;

            Console.Write(tableName);
            NewLine(posX);
            AppendTableRowSeparator(array[0].Length, posX);
            for (int i = 0; i < xLength; i++)
            {
                int yLength = array[i].Length;
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("|");
                Console.ForegroundColor = ConsoleColor.Green;
                Console.Write("{0:00}", i);
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("|");
                for (int j = 0; j < yLength; j++)
                {
                    SetConsoleColor(array[i][j]);
                    Console.Write("{0:0.00}", array[i][j]);
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write("|");
                }
                NewLine(posX);
                AppendTableRowSeparator(yLength, posX);
            }
            
            return new Tuple<int, int>(4 + array[0].Length * 5 + posX, array.Length * 2 + 2 + posY);
        }

        private static void SetConsoleColor(double d)
        {
            if (d < 0.25)
            {
                Console.ForegroundColor = ConsoleColor.Blue;
                return;
            }
            if (d < 0.5)
            {
                Console.ForegroundColor = ConsoleColor.DarkGreen;
                return;
            }
            if (d < 0.75)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                return;
            }
            if (d <= 1)
            {
                Console.ForegroundColor = ConsoleColor.Red;
            }
        }

        private static void AppendTableRowSeparator(int length, int posX)
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("+--+");
            for (int i = 0; i < length; i++)
            {
                Console.Write("----+");
            }
            NewLine(posX);
        }

        private static void NewLine(int posX)
        {
            Console.WriteLine("");
            int y = Console.CursorTop;
            Console.SetCursorPosition(posX, y);
        }
    }
}
