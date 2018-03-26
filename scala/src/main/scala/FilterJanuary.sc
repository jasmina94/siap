import java.io.{File, InputStream, PrintWriter}

import scala.io.Source
import java.net._

import scala.io

def CSVDemo(): Unit = {
  var stream: Iterator[String] = Source.fromResource("airports.csv").getLines()

  //  val url = new URL("file:///Users/zlatan/Do" +
  //    "wnloads/flights.csv")
  //
  //
  val writer = new PrintWriter(new File("/Users/zlatan/Desktop/january.csv"))
  //  val bufferedSource = io.Source.fromURL(url)
  var count = 0;

  //  bufferedSource.getLines.foreach { line =>
  //    val cols = line.split(",").map(_.trim)
  //
  //    if(cols(1).toString.equals("1")){
  //      writer.write(line + "," + "DATE")
  //      writer.write(System.getProperty("line.separator"))
  //      count += 1
  //    }
  //  }

  stream.foreach { line =>
    val cols = line.split(",").map(_.trim)

    if (cols(1).toString.equals("1")) {
      writer.write(line + "," + "DATE")
      writer.write(System.getProperty("line.separator"))
      count += 1
    }

    println(count)
    writer.close()
  }
}

CSVDemo()
