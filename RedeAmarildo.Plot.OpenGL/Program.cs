using System;

namespace RedeAmarildo.Plot.OpenGL
{
    /// <summary>
    /// The main class.
    /// </summary>
    public static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Console.WriteLine("Iris: ");
            Console.WriteLine(Algoritmo.algoritmoIris);
            Console.WriteLine("-----------------------------\n\n");

            Console.WriteLine("Custom: ");
            Console.WriteLine(Algoritmo.algoritmoCustom);
            Console.WriteLine();

            using (var game = new Game1())
                game.Run();
        }
    }
}
