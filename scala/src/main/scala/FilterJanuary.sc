import java.io.{File, PrintWriter}

import java.net._

def CSVDemo(): Unit = {

  val url = new URL("file:///Users/zlatan/Downloads/flights.csv")


  val writer = new PrintWriter(new File("/Users/zlatan/Desktop/january.csv"))
  val bufferedSource = io.Source.fromURL(url)
  var count = 0;

  bufferedSource.getLines.foreach { line =>
    val cols = line.split(",").map(_.trim)

    //cols(1).toString.equals("1") "1" is january, "2" is february...

    if (cols(1).toString.equals("1")) {
      writer.write(line + "," + "DATE")
      writer.write(System.getProperty("line.separator"))
      count += 1
    }
  }

  println(count)
  writer.close()
}


CSVDemo()
