namespace RedeAmarildo

open System
open FSharp.Data
open MathNet.Numerics
open MathNet.Numerics.Random
open MathNet.Numerics.LinearAlgebra
open System.Diagnostics

module Algoritmo =
    //Tipos
    type Par = { X: float Matrix; Y: float Matrix }
    type Realizacao = { TaxaAcerto:float; Confusao: float Matrix; W: float Matrix }
    type ResultadoAlgoritmo = { Acuracia: float; Tempo: TimeSpan; Melhor: Realizacao; Dados: Par seq; }

    //Utilitários
    let matrizLinha list =
        let v = (list: float seq) |> vector
        v.ToRowMatrix()

    let naoZero m =
        let zero = DenseMatrix.zero<float> (m: float Matrix).RowCount m.ColumnCount
        m <> zero

    //Funções Rede Perceptron
    let degrau u = 
        if u > 0.0 then 1.0 else 0.0

    let saida w x =
        x * w |> Matrix.map degrau

    let erro w x y =
        y - saida w x

    //Matriz de pesos para a matriz linha "treinamento"
    let pesos treinamento =
        (treinamento: Par list) |> ignore

        //Máximo de épocas
        let maxN = 2000

        //Próximo vetor de pesos (função w(n+1))
        let rec proximo t w e = 
            match t with
                | [] -> (w, e)
                | par :: tail -> 
                    let e0 = erro w par.X par.Y
                    let w1 = w + 0.05 * par.X.Transpose() * e0
                    let temErro = e || (e0 |> naoZero)

                    proximo tail w1 temErro
    
        //Decide se os pesos ainda devem ser atualizados (por número de épocas e ausência de erros)
        let rec pesos w n =
            let (w1, e1) = proximo (treinamento.SelectPermutation() |> List.ofSeq) w false
            if e1 && n < maxN  then pesos w1 (n+1) else w1
    
        let w0 = DenseMatrix.randomStandard<float> treinamento.Head.X.ColumnCount 3

        //Inicia o treinamento
        pesos w0 0


    let realizacao dados =
        let confusao = DenseMatrix.zero 3 3
    
        let treinamento = 
            let n = dados |> List.length |> float |> (*) 0.8 |> int
            dados |> List.take n

        let teste = dados |> List.except treinamento

        let w = pesos treinamento

        let classes = dict[[1.0; 0.0; 0.0] |> matrizLinha, 0; [0.0; 1.0; 0.0] |> matrizLinha, 1; [0.0; 0.0; 1.0] |> matrizLinha, 2]

        teste |>
            Seq.iter (fun par -> 
                let a = saida w par.X
                if classes.ContainsKey a then (confusao.[classes.[a], classes.[par.Y]] <- confusao.[classes.[a], classes.[par.Y]] + 1.0)
                )
        
        { TaxaAcerto = confusao.Diagonal().Sum() / float (teste |> Seq.length) ; Confusao = confusao; W = w }

    let sw = new Stopwatch()

    let algoritmoIris =
        sw.Start()
        let db = CsvFile.Load("iris.data").Cache()
        let classes = dict["Iris-setosa", [1.0; 0.0; 0.0]; "Iris-versicolor", [0.0; 1.0; 0.0]; "Iris-virginica", [0.0; 0.0; 1.0]]
    
        let parse s = s |> System.Double.Parse

        let parseRow (row: CsvRow) = row.Columns |> Seq.take 4 |> Seq.map parse |> List.ofSeq

        let mapRow (row: CsvRow) = { X = 1.0 :: parseRow row |> matrizLinha; Y = classes.[row.["class"]] |> matrizLinha }
    
        let dados = db.Rows |> Seq.map mapRow |> List.ofSeq

        let realizacoes =
            [1..20] |>
            Seq.map (fun _ -> realizacao (dados.SelectPermutation() |> List.ofSeq))
    
        let maior = 
            realizacoes |>
            Seq.maxBy (fun r -> r.TaxaAcerto)
        
        let media =
            realizacoes |>
            Seq.averageBy (fun r -> r.TaxaAcerto)
        
        //printfn "%A %A" media maior

        sw.Stop()
        { Acuracia = media; Tempo = sw.Elapsed; Melhor = maior; Dados = dados; }



    let algoritmoCustom =
        sw.Restart()
        let samples = 50
        let mapping x y =
            [1.0 ; x; y] |> matrizLinha

        let classe1 =
            let x = Random.doubles samples |> Seq.map (fun n -> n + 1.0)
            let y = Random.doubles samples |> Seq.map (fun n -> n + 1.0)
            Seq.map2 mapping x y |>
            Seq.map (fun x -> {X = x; Y = matrizLinha [1.0; 0.0; 0.0] }) |>
            List.ofSeq
    
        let classe2 =
            let x = Random.doubles samples |> Seq.map (fun n -> n + 3.0)
            let y = Random.doubles samples |> Seq.map (fun n -> n + 1.0)
            Seq.map2 mapping x y |>
            Seq.map (fun x -> {X = x; Y = matrizLinha [0.0; 1.0; 0.0] }) |>
            List.ofSeq

        let classe3 =
            let x = Random.doubles samples |> Seq.map (fun n -> n + 1.0)
            let y = Random.doubles samples |> Seq.map (fun n -> n + 3.0)
            Seq.map2 mapping x y |>
            Seq.map (fun x -> {X = x; Y = matrizLinha [0.0; 0.0; 1.0] }) |>
            List.ofSeq
        
    
        let dados = classe1 @ classe2 @ classe3

        let realizacoes =
            [1..20] |>
            Seq.map (fun _ -> realizacao (dados.SelectPermutation() |> List.ofSeq))
    
        let maior = 
            realizacoes |>
            Seq.maxBy (fun r -> r.TaxaAcerto)

        let media =
            realizacoes |>
            Seq.averageBy(fun r -> r.TaxaAcerto)
    
        sw.Stop()
        { Acuracia = media; Tempo = sw.Elapsed; Melhor = maior; Dados = dados; }