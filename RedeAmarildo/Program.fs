open FSharp.Data
open MathNet.Numerics
open MathNet.Numerics.LinearAlgebra
open System.Diagnostics

// Saiba mais sobre F# em http://fsharp.org
// Veja o projeto 'F# Tutorial' para obter mais ajuda.

type Par = { X: float Matrix; Y: float Matrix }
type Realizacao = { Acuracia:float; Confusao: float Matrix; W: float Matrix; Dados: Par list }

//Função degrau
let degrau u = 
    if u > 0.0 then 1.0 else 0.0

//Saída linear na forma de matriz
let saida w x =
    x * w |> Matrix.map degrau

let erro w x y =
    y - saida w x
    
let matrizLinha list =
    let v = (list: float seq) |> vector
    v.ToRowMatrix()

let naoZero m =
    (m: float Matrix) |> ignore
    let zero = DenseMatrix.zero<float> m.RowCount m.ColumnCount
    m <> zero

let pesos treinamento =
    let maxN = 1000
    let rec proximo t w e = 
        match t with
            | [] -> (w, e)
            | par :: tail -> 
                let e0 = erro w par.X par.Y
                let w1 = w + 0.01 * par.X.Transpose() * e0
                let temErro = 
                    if e then e else e0 |> naoZero
                proximo tail w1 (if e then e else temErro)
    
    let rec pesos w n =
        let (w1, e1) = proximo treinamento w false
        if e1 && n < maxN  then pesos w1 (n+1) else w1
    
    let w0 = DenseMatrix.randomStandard<float> treinamento.Head.X.ColumnCount 3
        
    pesos w0 0


let realizacao dados =
    let confusao = DenseMatrix.zero 3 3

    let dadosList = dados |> List.ofSeq

    let treinamento = 
        let n = dadosList |> List.length |> float |> (*) 0.8 |> int
        dadosList |> List.take n
    let teste = dadosList |> List.except treinamento

    let w = pesos treinamento

    let classes = dict[[1.0; 0.0; 0.0] |> matrizLinha, 0; [0.0; 1.0; 0.0] |> matrizLinha, 1; [0.0; 0.0; 1.0] |> matrizLinha, 2]

    teste |>
        Seq.iter (fun par -> 
            let a = saida w par.X
            if classes.ContainsKey a then (confusao.[classes.[a], classes.[par.Y]] <- confusao.[classes.[a], classes.[par.Y]] + 1.0)
            )
        
    { Acuracia = confusao.Diagonal().Sum() / float (teste |> Seq.length) ; Confusao = confusao; Dados = dadosList; W = w }

let sw = new Stopwatch()

sw.Start()

let algoritmoIris =
    let db = CsvFile.Load("iris.data").Cache()
    let classes = dict["Iris-setosa", [1.0; 0.0; 0.0]; "Iris-versicolor", [0.0; 1.0; 0.0]; "Iris-virginica", [0.0; 0.0; 1.0]]
    
    let parse s = s |> System.Double.Parse

    let mapRow (row: CsvRow) = { X = row.Columns |> Array.take 4 |> Array.map parse |> matrizLinha; Y = classes.[row.["class"]] |> matrizLinha }
    
    let dados = db.Rows |> Seq.map mapRow

    let realizacoes =
        [1..20] |>
        Seq.map (fun _ -> realizacao (dados.SelectPermutation()))
    
    let maior = 
        realizacoes |>
        Seq.maxBy (fun r -> r.Acuracia)

    maior

sw.Stop()


[<EntryPoint>]
let main argv = 
    printfn "%A" algoritmoIris
    printfn "Tempo: %d ms" sw.ElapsedMilliseconds
    0 // retornar um código de saída inteiro
