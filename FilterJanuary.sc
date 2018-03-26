import java.io.{File, PrintWriter}
import java.net._

import scala.io

def  CSVDemo(): Unit = {
  val url = new URL("file:///Users/zlatan/Do" +
    "wnloads/flight-delays/january.csv")

  val writer = new PrintWriter(new File("/Users/zlatan/Desktop/january.csv"))
  val bufferedSource = io.Source.fromURL(url)
  var count = 0;

  bufferedSource.getLines.foreach { line =>
    val cols = line.split(",").map(_.trim)

    if(cols(0).toString.equals("YEAR")){
      writer.write(line + "," + "DATE")
      writer.write(System.getProperty("line.separator"))
    }

    if(count < 50) {
      if (cols(1).length < 3) {
        count += 1
        val date = cols(2) + "/" + cols(1).toInt + "/" + cols(0).toInt
        writer.write(line + "," + date)
        writer.write(System.getProperty("line.separator"))
      }
    }
  }
  println(count)
  writer.close()
  bufferedSource.close
}

CSVDemo()
